# ML Pipeline

Details on feature engineering, model training, and evaluation for the Diamond Price Prediction system.

Run the full pipeline with:

```bash
python main.py --train
```

---

## Pipeline Steps

```
gemstone.csv ──▶ Ingestion ──▶ Transformation ──▶ Training ──▶ Evaluation ──▶ Feature Importance
```

---

## 1 — Data Ingestion

**File**: `src/DiamondPricePrediction/components/Data_ingestion.py`

- Reads `data/gemstone.csv` (193,573 rows × 11 columns)
- Saves a verbatim copy to `Artifacts/raw_data.csv`
- Splits 80 / 20 with `train_test_split(random_state=42)`
- Outputs `Artifacts/train_data.csv` and `Artifacts/test_data.csv`

---

## 2 — Data Transformation

**File**: `src/DiamondPricePrediction/components/Data_transformation.py`

### Feature layout

| Type | Features | Treatment |
|------|----------|-----------|
| Numerical | carat, depth, table, x, y, z | `SimpleImputer(median)` → `StandardScaler` |
| Categorical | cut, color, clarity | `SimpleImputer(most_frequent)` → `OrdinalEncoder` → `StandardScaler` |
| Dropped | id, price (target) | — |

### Ordinal encoding order

These orderings reflect the GIA grading scale:

| Feature | Low → High |
|---------|-----------|
| **Cut** | Fair → Good → Very Good → Premium → Ideal |
| **Color** | D → E → F → G → H → I → J |
| **Clarity** | I1 → SI2 → SI1 → VS2 → VS1 → VVS2 → VVS1 → IF |

Both pipelines are combined into a single `ColumnTransformer` and the fitted object is saved to `Artifacts/preprocessor.pkl` for reuse at inference.

---

## 3 — Model Training

**File**: `src/DiamondPricePrediction/components/Model_trainer.py`

### Models compared

| # | Model | Key Params | Category |
|---|-------|-----------|----------|
| 1 | Linear Regression | default | Linear |
| 2 | Lasso | alpha = 1.0 | L1 regularized |
| 3 | Ridge | alpha = 1.0 | L2 regularized |
| 4 | ElasticNet | default | L1 + L2 |
| 5 | Decision Tree | random_state = 42 | Tree-based |
| 6 | Random Forest | n_estimators = 100, n_jobs = -1 | Bagging ensemble |
| 7 | Gradient Boosting | n_estimators = 100 | Boosting ensemble |

### Training process

1. Fit each model on the full training array
2. Score each on the held-out test array (R²)
3. Run 5-fold cross-validation on the training set for each model
4. Log all metrics; save per-model results to `Artifacts/model_evaluation_results.json`
5. Select the model with the highest **test R²** → save as `Artifacts/model.pkl`

---

## 4 — Evaluation

**File**: `src/DiamondPricePrediction/components/Model_evaluation.py`

The final selected model is evaluated on both sets and three metrics are recorded:

| Metric | What it measures |
|--------|-----------------|
| **R²** | Proportion of variance explained (1.0 = perfect) |
| **MAE** | Mean absolute error in USD |
| **RMSE** | Root mean squared error in USD (penalizes large errors) |

### Current results (Random Forest)

| Metric | Train | Test |
|--------|------:|-----:|
| R² | 0.997 | 0.977 |
| MAE | $116 | $309 |
| RMSE | $228 | $607 |

### Full model comparison

| Model | Test R² | Test MAE | Test RMSE | CV R² (mean ± std) |
|-------|--------:|---------:|----------:|--------------------:|
| **Random Forest** | **0.977** | **$309** | **$607** | 0.977 ± 0.001 |
| Gradient Boosting | 0.976 | $331 | $621 | 0.977 ± 0.001 |
| Decision Tree | 0.957 | $424 | $835 | 0.957 ± 0.001 |
| Linear Regression | 0.937 | $672 | $1,007 | 0.936 ± 0.001 |
| Lasso | 0.937 | $673 | $1,007 | 0.937 ± 0.001 |
| Ridge | 0.937 | $672 | $1,007 | 0.936 ± 0.001 |
| ElasticNet | 0.854 | $1,063 | $1,536 | 0.855 ± 0.001 |

Output files:

- `Artifacts/model_evaluation_results.json` — per-model metrics
- `Artifacts/final_model_metrics.json` — selected model train + test metrics

---

## 5 — Feature Importance

After training, importance scores are extracted from the best model:

- **Tree-based** models → `.feature_importances_` (Gini importance)
- **Linear** models → absolute coefficient values `|coef_|`

A bar chart is saved to `reports/feature_importance.png`.

Expected ranking (most → least important):

1. **carat** — primary price driver
2. **x, y, z** — physical dimensions (highly correlated with carat)
3. **clarity** — higher grades command premiums
4. **color** — better grades (D–F) increase value
5. **cut** — affects brilliance
6. **depth, table** — secondary geometric properties

---

## Prediction Pipeline

**File**: `src/DiamondPricePrediction/pipelines/Prediction_Pipeline.py`

Used at inference time when a user submits the web form:

1. `CustomData` accepts the 9 input fields and builds a single-row `DataFrame`
2. `PredictPipeline.predict(df)`:
   - Loads `Artifacts/preprocessor.pkl`
   - Transforms the input
   - Loads `Artifacts/model.pkl`
   - Returns the predicted price as a float

The Flask route in `app.py` rounds the result to two decimal places and renders it in `result.html`.

This pipeline is called by the Flask web app (`app.py`) when users submit the prediction form.

---

## Reproducibility

- All random operations use `random_state=42`
- Train/test split is deterministic
- DVC pipeline definition in `dvc.yaml` tracks data and code dependencies
- Evaluation results are saved as JSON for auditability
