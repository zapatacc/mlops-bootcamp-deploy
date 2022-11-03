import pandas as pd
from fastapi import FastAPI
from features import Features
import pickle
import uvicorn
import sklearn

app = FastAPI()


@app.on_event("startup")
def load_model():

    global model_fraud, scaler

    with open("./model/model_fraud.pickle", "rb") as openfile:
        model_fraud = pickle.load(openfile)

    with open("./model/scaler.pickle", "rb") as openfile:
        scaler = pickle.load(openfile)


@app.get("/api/v1/detectFraud")
def detect_fraud(features: Features):

    df_features = pd.DataFrame(features.__dict__, index=[0])

    cols_order = ['feature_06', 'feature_02', 'feature_05', 'feature_01',
                   'feature_23', 'feature_19', 'feature_10', 'amount', 'feature_22',
                   'feature_28', 'timestamp', 'feature_11', 'feature_21', 'feature_13',
                   'feature_17', 'feature_09', 'feature_25', 'feature_08',
                   'feature_26', 'feature_24', 'feature_15', 'feature_18',
                   'feature_27', 'feature_12', 'feature_20']

    df_features = df_features[cols_order]

    values_ = df_features.values

    pred = model_fraud.predict(scaler.transform(values_))

    return {"Fraudulent": int(pred[0])}


@app.get("/")
def home():
    return {"Desc": "Health Check"}


if __name__ == '__main__':
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)


