# Architecture

How the Diamond Price Prediction system is organized, what each component does, and how data flows from raw CSV to a predicted price in the browser.

---

## System Diagram

```
                TRAINING (offline, one-time)
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  data/gemstone.csv
       │
       ▼
  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
  │  Ingestion   │──▶│ Transform    │──▶│   Trainer    │──▶│  Evaluation  │
  │  80/20 split │   │ OrdinalEnc + │   │  7 models    │   │  R², MAE,    │
  │              │   │ StdScaler    │   │  5-fold CV   │   │  RMSE → JSON │
  └──────────────┘   └──────┬───────┘   └──────┬───────┘   └──────────────┘
                            │                  │
                            ▼                  ▼
                     preprocessor.pkl     model.pkl


                PREDICTION (live, per request)
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Browser (form.html)
       │  POST 9 features
       ▼
  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
  │  Flask route  │──▶│ CustomData   │──▶│ PredictPipe  │──▶ $price
  │  app.py       │   │ → DataFrame  │   │ preproc +    │
  └──────────────┘   └──────────────┘   │ model.pkl    │
                                         └──────────────┘
```

---

## Components

### Data Layer

| Component | File | Role |
|-----------|------|------|
| Raw dataset | `data/gemstone.csv` | 193,573 diamonds, 10 columns |
| Train split | `Artifacts/train_data.csv` | 80 % of data (random_state=42) |
| Test split | `Artifacts/test_data.csv` | 20 % held-out evaluation set |

### Processing Layer

| Component | File | Role |
|-----------|------|------|
| Data Ingestion | `components/Data_ingestion.py` | Reads CSV, saves raw copy, splits 80/20 |
| Data Transformation | `components/Data_transformation.py` | Ordinal encoding + standard scaling via `ColumnTransformer` |
| Preprocessor artifact | `Artifacts/preprocessor.pkl` | Fitted transformer reused at inference |

### Model Layer

| Component | File | Role |
|-----------|------|------|
| Model Trainer | `components/Model_trainer.py` | Trains 7 regressors, 5-fold CV, selects best by test R² |
| Model Evaluation | `components/Model_evaluation.py` | Evaluates on train + test sets, writes JSON |
| Model artifact | `Artifacts/model.pkl` | Serialized Random Forest (best model) |
| Per-model metrics | `Artifacts/model_evaluation_results.json` | R², MAE, RMSE, CV stats for all 7 models |
| Final metrics | `Artifacts/final_model_metrics.json` | Train + test metrics for selected model |

### Serving Layer

| Component | File | Role |
|-----------|------|------|
| Flask app | `app.py` | Single route `/` — GET renders form, POST returns result |
| Prediction Pipeline | `pipelines/Prediction_Pipeline.py` | Loads preprocessor + model, transforms input, predicts |
| Form template | `templates/form.html` | 3 fieldsets, input validation, loading overlay |
| Result template | `templates/result.html` | Animated price counter, input summary, model info card |
| Stylesheet | `static/style.css` | Dark theme, CSS custom properties, responsive breakpoints |

### Infrastructure

| Component | File | Role |
|-----------|------|------|
| CLI entry point | `main.py` | `--train`, `--serve`, or both |
| Logger | `src/.../logger.py` | Timestamped file + console logs in `logs/` |
| Exception handler | `src/.../exception.py` | Custom exception with file + line info |
| Dockerfile | `Dockerfile` | Container definition |


---

## Data Flow: Training

```
data/gemstone.csv
    │
    ▼
DataIngestion.initiate_data_ingestion()
    ├── Artifacts/raw_data.csv
    ├── Artifacts/train_data.csv   (80 %)
    └── Artifacts/test_data.csv    (20 %)
         │
         ▼
DataTransformation.initialize_data_transformation()
    ├── Fits OrdinalEncoder on cut, color, clarity
    ├── Fits StandardScaler on all features
    ├── Saves Artifacts/preprocessor.pkl
    └── Returns train_arr, test_arr (NumPy arrays)
         │
         ▼
ModelTrainer.initate_model_training()
    ├── Trains 7 regression models
    ├── 5-fold cross-validates each
    ├── Selects best by test R²
    ├── Saves Artifacts/model.pkl
    └── Saves Artifacts/model_evaluation_results.json
         │
         ▼
ModelEvaluation.initiate_model_evaluation()
    ├── Loads model.pkl
    ├── Evaluates on full train & test arrays
    └── Saves Artifacts/final_model_metrics.json

Training_pipeline.run_training_pipeline()
    └── Plots feature importance → reports/feature_importance.png
```

## Data Flow: Prediction

```
User submits form (9 fields)
    │
    ▼
CustomData.get_data_as_dataframe()
    │  Builds a single-row DataFrame
    ▼
PredictPipeline.predict(df)
    ├── Loads Artifacts/preprocessor.pkl
    ├── Transforms input features
    ├── Loads Artifacts/model.pkl
    └── Returns predicted price (float)
         │
         ▼
Flask renders result.html with $price
    │  sessionStorage passes form inputs
    ▼
Result page shows animated price + input summary
```

---

## Logging

All components use Python's built-in `logging` module:

- **Format**: `[timestamp] module — LEVEL — message`
- **Outputs**: timestamped log files in `logs/` + console
- Covers: ingestion, transformation, training, predictions, errors

---

## Deployment Options

| Method     | Command                        | URL                    |
|------------|--------------------------------|------------------------|
| Local      | `python main.py`               | http://localhost:8080  |
| Docker     | `docker run -p 8080:8080 img`  | http://localhost:8080  |
