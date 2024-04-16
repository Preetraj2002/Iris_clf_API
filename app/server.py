from fastapi import FastAPI
import joblib
import numpy as np

model = joblib.load("app/iris_svm.joblib")

class_name = ["setosa", "versicolor", "virginica"]

app = FastAPI()

@app.get('/')
def root():
    return {"message":"Welcome to Iris classification API"}


@app.post('/predict')
def predict(data: dict):
    features = np.array(data['features']).reshape(1, -1)
    prediction = model.predict(features)
    pred_class = class_name[prediction[0]]
    return {"predicted_class":pred_class}

predict({"features":[1,2,3,4]})