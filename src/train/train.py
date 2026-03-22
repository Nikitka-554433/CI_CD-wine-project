import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import joblib
import os

# Папка для модели
os.makedirs("models", exist_ok=True)

# Загрузка данных
df = pd.read_csv("data/raw/wine.csv")
X = df.drop("quality", axis=1)
y = df["quality"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# LinearRegression
with mlflow.start_run(run_name="LinearRegression"):
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    mlflow.log_param("model_type", "LinearRegression")
    mlflow.log_metric("mae", mean_absolute_error(y_test, y_pred))
    mlflow.log_metric("r2", r2_score(y_test, y_pred))
    mlflow.sklearn.log_model(model, "model")
    print(f"LinearRegression - MAE: {mean_absolute_error(y_test, y_pred):.3f}")

#RandomForest
with mlflow.start_run(run_name="RandomForest"):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    mlflow.log_param("model_type", "RandomForest")
    mlflow.log_param("n_estimators", 100)
    mlflow.log_metric("mae", mean_absolute_error(y_test, y_pred))
    mlflow.log_metric("r2", r2_score(y_test, y_pred))
    mlflow.sklearn.log_model(model, "model")
    
    # Сохранение лучшей модели
    joblib.dump(model, "models/model.pkl")
    print(f" {mean_absolute_error(y_test, y_pred):.3f}")

print(" Лучшая модель сохранена в models/model.pkl")