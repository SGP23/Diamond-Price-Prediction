<div align="center">

# 💎 Diamond Price Prediction

**An end-to-end machine learning system that predicts diamond prices from physical and quality characteristics.**

Trained on 193,573 diamonds &bull; 7 models compared &bull; Best R² = 0.976

![Python 3.8+](https://img.shields.io/badge/python-3.8+-3776ab?style=flat-square&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-f7931e?style=flat-square&logo=scikit-learn&logoColor=white)
![Flask](https://img.shields.io/badge/flask-2.x-000000?style=flat-square&logo=flask&logoColor=white)
![License: MIT](https://img.shields.io/badge/license-MIT-green?style=flat-square)

</div>

---

## Overview

This project builds a complete ML pipeline — from raw data to a web application you can interact with. A user enters nine diamond characteristics in a form, and the model returns a predicted market price in **Indian Rupees (₹)**.

> The ML model predicts prices in USD internally. The application converts the output to INR using a fixed rate of **1 USD = 92 INR**.

**Highlights**

- Compares **7 regression algorithms** (Linear, Lasso, Ridge, ElasticNet, Decision Tree, Random Forest, Gradient Boosting)
- Selects the best model via **5-fold cross-validation** on R², MAE, and RMSE
- Generates a **feature importance** visualization
- Serves predictions through a **Flask web app** with a dark-themed, glassmorphism UI
- Structured logging, reproducible artifacts, and Docker support

---

## Demo

```
              ┌─────────────────┐         ┌───────────────────┐
  Browser ──▶ │  Flask (app.py) │ ──▶     │ PredictPipeline   │
              │  GET  → form    │         │  preprocessor.pkl │
              │  POST → result  │ ◀── ₹   │  model.pkl        │
              └─────────────────┘         └───────────────────┘
```

Enter diamond specs → click **Predict Price** → see animated result in ₹ (INR) with model details.

---

## Model Performance

> Best model: **Random Forest Regressor** (100 trees)

| Metric | Train | Test |
|--------|------:|-----:|
| R²     | 0.997 | 0.977 |
| MAE    | $116  | $309  |
| RMSE   | $228  | $607  |

<details>
<summary>Full comparison across all 7 models</summary>

| Model | Test R² | Test MAE | Test RMSE | CV R² (mean ± std) |
|-------|--------:|---------:|----------:|--------------------:|
| Random Forest | **0.977** | **$309** | **$607** | 0.977 ± 0.001 |
| Gradient Boosting | 0.976 | $331 | $621 | 0.977 ± 0.001 |
| Decision Tree | 0.957 | $424 | $835 | 0.957 ± 0.001 |
| Linear Regression | 0.937 | $672 | $1,007 | 0.936 ± 0.001 |
| Lasso | 0.937 | $673 | $1,007 | 0.937 ± 0.001 |
| Ridge | 0.937 | $672 | $1,007 | 0.936 ± 0.001 |
| ElasticNet | 0.854 | $1,063 | $1,536 | 0.855 ± 0.001 |

</details>

---

## Dataset

~193,000 diamonds from the [Kaggle Playground Series S3E8](https://www.kaggle.com/competitions/playground-series-s3e8/data).

| Feature | Description | Type |
|---------|-------------|------|
| carat | Weight | Numerical |
| cut | Cut quality (Fair → Ideal) | Ordinal |
| color | Color grade (D → J) | Ordinal |
| clarity | Clarity grade (I1 → IF) | Ordinal |
| depth | Total depth percentage | Numerical |
| table | Top facet width (%) | Numerical |
| x, y, z | Length, width, height (mm) | Numerical |
| **price** | **Target — USD** | **Numerical** |

---

## ML Pipeline

```
gemstone.csv ──▶ Ingestion ──▶ Transformation ──▶ Training ──▶ Evaluation
                  (80/20)      (OrdinalEnc +      (7 models,    (R², MAE,
                                StandardScaler)    5-fold CV)    RMSE → JSON)
```

1. **Data Ingestion** — loads CSV, saves raw copy, performs 80/20 train/test split
2. **Data Transformation** — ordinal-encodes cut/color/clarity, standard-scales all features, saves `preprocessor.pkl`
3. **Model Training** — trains 7 regressors, cross-validates, picks best by test R², saves `model.pkl`
4. **Model Evaluation** — evaluates final model on held-out test set, writes metrics to JSON
5. **Feature Importance** — extracts and plots importances to `reports/feature_importance.png`

---

## Project Structure

```
├── main.py                        # CLI entry point (--train / --serve)
├── app.py                         # Flask web server (port 8080)
├── requirements.txt
├── setup.py
├── Dockerfile
│
├── data/
│   └── gemstone.csv               # Source dataset (193 K diamonds)
│
├── src/DiamondPricePrediction/
│   ├── components/
│   │   ├── Data_ingestion.py
│   │   ├── Data_transformation.py
│   │   ├── Model_trainer.py
│   │   └── Model_evaluation.py
│   ├── pipelines/
│   │   ├── Training_pipeline.py
│   │   └── Prediction_Pipeline.py
│   ├── utils/utils.py
│   ├── logger.py
│   └── exception.py
│
├── templates/
│   ├── form.html                  # Prediction form (glassmorphism UI)
│   └── result.html                # Result page (animated price counter)
├── static/style.css               # Design system (CSS custom properties)
│
├── notebooks/EDA.ipynb
│
├── Artifacts/                     # Generated by training (gitignored)
├── reports/                       # Generated plots (gitignored)
├── logs/                          # Runtime logs (gitignored)
└── docs/
    ├── architecture.md
    └── ml_pipeline.md
```

---

## Getting Started

### Prerequisites

- Python 3.8+
- pip

### Installation

```bash
git clone https://github.com/SGP23/Diamond-Price-Prediction.git
cd Diamond-Price-Prediction

python -m venv venv
# Linux / macOS
source venv/bin/activate
# Windows
venv\Scripts\activate

pip install -r requirements.txt
```

### Usage

```bash
# Train the model and start the web server
python main.py

# Or individually:
python main.py --train      # Run training pipeline only
python main.py --serve      # Start web server only (trains first if no model found)
```

Then open **http://localhost:8080**.

### Docker

```bash
docker build -t diamond-price-prediction .
docker run -p 8080:8080 diamond-price-prediction
```

---

## Web Interface

The UI is a single-page dark-themed design built with vanilla HTML/CSS/JS:

- **Form page** — grouped fieldsets (Physical, Dimensions, Quality), unit indicators, input validation, loading overlay with diamond animation
- **Result page** — animated price counter, input summary table, model info card, fade-up entrance animations
- **Responsive** — breakpoints at 768 px and 480 px

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| ML | scikit-learn |
| Data | pandas, NumPy |
| Visualization | matplotlib, seaborn |
| Web | Flask, Jinja2 |
| Frontend | HTML5, CSS3 (custom properties, glassmorphism), vanilla JS |
| Containerization | Docker |
| Logging | Python `logging` (file + console) |

---

## Documentation

- [docs/architecture.md](docs/architecture.md) — system components and data flow
- [docs/ml_pipeline.md](docs/ml_pipeline.md) — feature engineering, model training, and evaluation details

---

## Currency Conversion

The ML model is trained on diamond prices in **USD** and all internal predictions remain in USD. The application layer converts the predicted price to **Indian Rupees (₹)** before displaying it to the user.

| Parameter | Value |
|-----------|-------|
| Internal prediction currency | USD |
| Display currency | INR (₹) |
| Fixed conversion rate | **1 USD = 92 INR** |
| Conversion location | `app.py` (after model prediction) |

The conversion rate is a fixed constant set by the developer and is **not** fetched from an external API.

---

## License

MIT — see [LICENSE](LICENSE).
