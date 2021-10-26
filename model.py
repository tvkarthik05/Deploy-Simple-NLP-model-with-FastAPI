import pandas as pd
import numpy as np
import gensim
import pickle
import warnings

from gensim.models import Word2Vec
from gensim.parsing.preprocessing import remove_stopwords
from gensim.utils import simple_preprocess
from nltk.tokenize import sent_tokenize, word_tokenize
from pydantic import BaseModel
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from gensim.parsing.porter import PorterStemmer


class data_load(BaseModel):
    data: str

class inf_check(BaseModel):
    text: str 
    

def pre_processing(record):
    porter_stemmer = PorterStemmer()
    t0 = simple_preprocess(record, deacc=True)
    t1 = [porter_stemmer.stem(word) for word in t0]
    return t1

class w2v_model:
    def __init__(self):
        return None
    
    def train(self, data):
        self.df = pd.read_csv(data, nrows=5000)
        print(self.df)
        print('Data loaded')

        self.df = self.df.loc[(self.df['Consumer complaint narrative'].notnull()), ['Consumer complaint narrative', 'Product']].reset_index().drop('index', axis = 1)
        self.df.columns = ['text', 'target']
        print('Dataframe size - ', self.df.shape)
        
        self.df['stemmed_tokens'] = [pre_processing(line) for line in self.df['text']]
        
        # Train Test Split Function
        X_train, X_test, Y_train, Y_test = train_test_split(self.df.drop(['target'], axis=1), 
                                                            self.df['target'], 
                                                            test_size=0.2, 
                                                            random_state=15)

        print('Train shape - ', X_train.shape)
        print('Test shape - ', X_test.shape)

        # Skip-gram model (sg = 1)
        window = 3
        min_count = 1
        workers = 3
        sg = 1

        word2vec_model_file = 'word2vec_model.model'
        stemmed_tokens = pd.Series(self.df['stemmed_tokens']).values
        # Train the Word2Vec Model
        w2v_model = Word2Vec(stemmed_tokens, min_count = min_count, workers = workers, window = window, sg = sg)
        # building vocabulary for training
        w2v_model.build_vocab(stemmed_tokens)
        # reducing the epochs will decrease the computation time
        print('Training started!!!')
        w2v_model.train(stemmed_tokens, total_examples=len(stemmed_tokens), epochs=2)
        w2v_model.save('./models/'+word2vec_model_file)

        # Store the vectors for train data in following file
        word2vec_filename = './models/word2vec.csv'
        with open(word2vec_filename, 'w+') as word2vec_file:
            for index, row in X_train.iterrows():
                model_vector = (np.mean([w2v_model.wv[token] for token in row['stemmed_tokens']], axis=0)).tolist()
                if index == 0:
                    header = ",".join(str(ele) for ele in range(100))
                    word2vec_file.write(header)
                    word2vec_file.write("\n")
                # Check if the line exists else it is vector of zeros
                if type(model_vector) is list:  
                    line1 = ",".join( [str(vector_element) for vector_element in model_vector] )
                else:
                    line1 = ",".join([str(0) for i in range(100)])
                word2vec_file.write(line1)
                word2vec_file.write('\n')

        
        # Load from the filename
        word2vec_df = pd.read_csv(word2vec_filename)
        #Initialize the model
        clf_decision_word2vec = DecisionTreeClassifier()

        # Fit the model
        clf_decision_word2vec.fit(word2vec_df, Y_train)
        
        with open('./models/dt_model.pkl','wb') as f:
            pickle.dump(clf_decision_word2vec,f)
        f.close()
        
        test_features_word2vec = []
        for index, row in X_test.iterrows():
            model_vector = np.mean([w2v_model.wv[token] for token in row['stemmed_tokens']], axis=0)
            if type(model_vector) is list:
                test_features_word2vec.append(model_vector)
            else:
                test_features_word2vec.append(np.array([0 for i in range(100)]))
        test_predictions_word2vec = clf_decision_word2vec.predict(test_features_word2vec)
        print(classification_report(Y_test,test_predictions_word2vec))

        print('Training completed successfully!!!')

    def inference(self, text):
        input = pre_processing(text)
        w2v_model = Word2Vec.load('./models/word2vec_model.model')
        text_vector = np.mean([w2v_model.wv[token] for token in input], axis=0)
        df_text_vector = pd.DataFrame(np.reshape(text_vector, (1,len(text_vector))))
        
        with open('./models/dt_model.pkl', 'rb') as f:
            model = pickle.load(f)
        f.close()

        
        pred = model.predict(df_text_vector)
        print('Inference completed successfully!!!')
        return pred

