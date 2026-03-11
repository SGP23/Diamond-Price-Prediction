import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
from src.DiamondPricePrediction.logger import logging
from src.DiamondPricePrediction.exception import customexception

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import cross_val_score


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
    except Exception as e:
        raise customexception(e, sys)


def evaluate_model(X_train, y_train, X_test, y_test, models):
    """Evaluate multiple models with cross-validation and comprehensive metrics."""
    try:
        report = {}
        detailed_report = {}

        for model_name, model in models.items():
            logging.info(f"Training model: {model_name}")

            # Train model
            model.fit(X_train, y_train)

            # Predict on train and test
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # Test metrics
            test_r2 = r2_score(y_test, y_test_pred)
            test_mae = mean_absolute_error(y_test, y_test_pred)
            test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

            # Train metrics
            train_r2 = r2_score(y_train, y_train_pred)

            # Cross-validation (5-fold)
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()

            # Store R2 for backward compatibility
            report[model_name] = test_r2

            # Store detailed metrics
            detailed_report[model_name] = {
                'train_r2': round(float(train_r2), 4),
                'test_r2': round(float(test_r2), 4),
                'test_mae': round(float(test_mae), 2),
                'test_rmse': round(float(test_rmse), 2),
                'cv_r2_mean': round(float(cv_mean), 4),
                'cv_r2_std': round(float(cv_std), 4),
            }

            logging.info(
                f"{model_name}: R2={test_r2:.4f}, MAE={test_mae:.2f}, "
                f"RMSE={test_rmse:.2f}, CV_R2={cv_mean:.4f}±{cv_std:.4f}"
            )

        # Save detailed report
        os.makedirs("Artifacts", exist_ok=True)
        report_path = os.path.join("Artifacts", "model_evaluation_results.json")
        with open(report_path, 'w') as f:
            json.dump(detailed_report, f, indent=2)
        logging.info(f"Detailed model evaluation saved to {report_path}")

        # Print comparison table
        print("\n" + "=" * 90)
        print(f"{'Model':<25} {'Train R²':<12} {'Test R²':<12} {'MAE':<12} {'RMSE':<12} {'CV R² (5-fold)'}")
        print("-" * 90)
        for name, metrics in detailed_report.items():
            print(
                f"{name:<25} {metrics['train_r2']:<12} {metrics['test_r2']:<12} "
                f"{metrics['test_mae']:<12} {metrics['test_rmse']:<12} "
                f"{metrics['cv_r2_mean']}±{metrics['cv_r2_std']}"
            )
        print("=" * 90 + "\n")

        return report

    except Exception as e:
        logging.info('Exception occurred during model training')
        raise customexception(e, sys)


def load_object(file_path):
    try:
        with open(file_path, 'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        logging.info('Exception Occurred in load_object function utils')
        raise customexception(e, sys)

    