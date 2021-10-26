# http://localhost:8000/docs - For testing API
# pip install fastapi uvicorn[standard]

import uvicorn
from fastapi import FastAPI
from model import inf_check, w2v_model, data_load
import pdb

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Word2Vec Model API"}

@app.post("/training")
def training(w2v: data_load):
    data = w2v.data
    model = w2v_model()
    training = model.train(data)
    
    return {"message": "Training completed"}

@app.post("/inference")
def inference(w2v: inf_check):
    text = w2v.text
    model = w2v_model()
    prediction = model.inference(text)
    print(prediction)
    return {"message": "Predicted Output - {}".format(prediction)}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)