import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os


app = FastAPI(title="Wine Quality Prediction API")

# Глобальная переменная для модели
model = None


# Загрузка модели
@app.on_event("startup")
async def load_model():
    global model
    model_path = "models/model.pkl"
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        print("Модель загружена")
    else:
        print("Модель не найдена, далее train.py")

# Описание входных данных
class WineFeatures(BaseModel):
    fixed_acidity: float
    volatile_acidity: float
    citric_acid: float
    residual_sugar: float
    chlorides: float
    free_sulfur_dioxide: float
    total_sulfur_dioxide: float
    density: float
    pH: float
    sulphates: float
    alcohol: float



# Эндпоинт для предсказания
@app.post("/predict")
async def predict(features: WineFeatures):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    input_dict = {
        "fixed acidity": features.fixed_acidity,
        "volatile acidity": features.volatile_acidity,
        "citric acid": features.citric_acid,
        "residual sugar": features.residual_sugar,
        "chlorides": features.chlorides,
        "free sulfur dioxide": features.free_sulfur_dioxide,
        "total sulfur dioxide": features.total_sulfur_dioxide,
        "density": features.density,
        "pH": features.pH,
        "sulphates": features.sulphates,
        "alcohol": features.alcohol
    }
    input_df = pd.DataFrame([input_dict])
    prediction = model.predict(input_df)[0]
    return {"predicted_quality": round(prediction, 2)}

@app.get("/healthcheck")
async def healthcheck():
    if model is not None:
        return {"status": "ok"}
    return {"status": "error", "reason": "model"}

# модель
@app.get("/model-info")
async def model_info():
    return {
        "model_type": "RandomForest (ожидается)",
        "version": "v1",
        "status": "loaded" if model is not None else "not loaded"
    }