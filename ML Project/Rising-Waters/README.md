# Rising Waters – AI Flood Risk Predictor

A premium AI SaaS-style experience for estimating flood risk using rainfall history and climate signals. This project combines a full machine-learning pipeline (data prep, training, evaluation) with a polished Bootstrap-based glassmorphism frontend and a Flask inference API.

## Highlights
- 📊 End-to-end pipeline: load Excel data, clean, feature engineer, preprocess, train multiple models, and pick the best performer.
- 🧠 Models: Logistic Regression, Random Forest, Gradient Boosting, XGBoost (best model saved), plus an optional LSTM prototype for sequential rainfall patterns.
- 🧹 Robust preprocessing: imputers, scaling, one-hot encoding, and feature selection tuned for inference stability.
- 🌐 Serving: Flask app with a premium UI, dark/light toggle, animated background, and responsive layout; REST endpoint for programmatic use.
- 📈 Metrics & artifacts: training metrics stored in `models/training_results.json`, best model in `models/flood_model.pkl`, preprocessor/scaler persisted for inference, evaluation plots saved to `models/`.

## Project structure
- `app.py` – Flask server with web UI and `/api/predict` endpoint.
- `templates/index.html` – Premium Bootstrap glass UI with animations, loaders, and sample-fill helpers.
- `src/data_loader.py` – Reads and cleans Excel datasets.
- `src/feature_engineering.py` – Merges rainfall + flood data, builds engineered features (lags, rolling means, deviations, climate covariates).
- `src/preprocessing.py` – Imputers, scaling, encoding, train/test split, and artifact persistence.
- `src/train.py` – Trains four classifiers, scores them, and saves the best model + metrics JSON.
- `src/evaluate.py` – Generates reports and plots (confusion matrix, ROC, feature importance).
- `src/predict.py` – Loads artifacts, aligns inference schema, and returns risk + probability.
- `src/lstm_model.py` – Optional LSTM experiment for sequential rainfall windows.
- `dataset/` – Input Excel files (see Data section).
- `models/` – Outputs (model, preprocessor, scaler, metrics, plots).

## Data
Place the provided Excel files inside `dataset/`:
- `flood dataset.xlsx`
- `rainfall in india 1901-2015.xlsx`

The pipelines expect these filenames by default. Update paths in `src/data_loader.py` if you change them.

## Setup
1. Install Python 3.10+ (project tested on 3.12).
2. (Optional) create a virtual environment:
   - Windows: `python -m venv .venv && .venv\Scripts\activate`
3. Install dependencies: `pip install -r requirements.txt`

## Training
Run the classical models and persist artifacts:
```
python -m src.train
```
Outputs:
- `models/flood_model.pkl` – best-performing model
- `models/preprocessor.joblib` & `models/scaler.joblib`
- `models/training_results.json` – metrics (accuracy, F1, ROC-AUC) and best model name

## Evaluation
Generate evaluation plots and feature importance:
```
python -m src.evaluate
```
Artifacts are saved to `models/` (confusion matrix, ROC curve, feature importance chart/CSV).

## Optional LSTM experiment
Train a simple LSTM over rainfall sequences:
```
python -m src.lstm_model
```
Model is saved to `models/flood_lstm.h5`.

## Running the app
Start the Flask server (defaults to port 5000):
```
python app.py
```
Then open http://127.0.0.1:5000/

UI perks: animated gradient background, frosted-glass cards, dark/light toggle, floating particles, wave divider, ripple buttons, loading spinner on predict, circular probability indicator, and progress bar. A sample-fill button lets you try random values quickly.

### API usage
POST JSON to `/api/predict`:
```
curl -X POST http://127.0.0.1:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
        "state": "Maharashtra",
        "year": 2015,
        "rainfall": 1200,
        "previous_year_rainfall": 1100,
        "avgjune": 210,
        "temp": 30,
        "humidity": 75,
        "cloud_cover": 40,
        "sub": 450
      }'
```
Response:
```
{"risk": "High", "probability": 0.72}
```

## Notes & tips
- If you change data schemas, revisit `feature_engineering.py` keep/drop lists and re-run training.
- Missing artifacts? Run `python -m src.train` to regenerate `flood_model.pkl` and preprocessors.
- Set `PORT` env var to change the serving port (optional).
- GPU/TF is optional; the core pipeline uses scikit-learn/XGBoost.

## License
Add your preferred license here.
