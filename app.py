import uvicorn
from fastapi import FastAPI
from BankNotes import BankNote
import numpy as np
import pickle
import pandas as pd

app=FastAPI()
pickle_in = open(r"C:\Users\JISHAN AICH\PycharmProjects\Fastapi\.venv\classifier.pkl", "rb")
classifier = pickle.load(pickle_in)

@app.get("/")
def index():
    return {"message": "Welcome to FastAPI!, replace the last part of the url with doc to open the api"}

@app.get("/{name}")
def get_name(name: str):
    return {'Hey! ': f'{name} this is a demo of FastAPI!'}

@app.post("/predict")
def predict_banknote(data:BankNote):
    data = data.dict()
    variance=data["variance"]
    skewness=data["skewness"]
    curtosis=data["curtosis"]
    entropy=data["entropy"]
    prediction=classifier.predict([[variance,skewness,curtosis,entropy]])
    if(prediction[0]>0.5):
        prediction="It is a Fake Note"
    else:
        prediction="It is a Bank Note"
    return {"prediction":prediction}

if __name__ == "__main__":
    uvicorn.run(app, host="127 .0.0.1", port=8000)
#uvicorn app:app --reload