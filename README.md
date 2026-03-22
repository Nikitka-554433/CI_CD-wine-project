# MLOps проект

# Описание
Сервис предсказания качества вина

# Структура проекта
- `src/api/` — FastAPI приложение
- `tests/` — тесты API
- `data/` — данные (версионируются через DVC)
- `models/` — сохранённые модели

# Запуск API командой

uvicorn src.api.app:app --reload --port 8000

pytest tests/ -v
mlflow ui --port 5000