import os
import sys
import json
import numpy as np
import logging
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from src.DiamondPricePrediction.utils.utils import load_object

logger = logging.getLogger(__name__)


class ModelEvaluation:
    def __init__(self):
        pass

    def eval_metrics(self, actual, pred):
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)
        return rmse, mae, r2

    def initiate_model_evaluation(self, train_array, test_array):
        try:
            X_test, y_test = (test_array[:, :-1], test_array[:, -1])
            X_train, y_train = (train_array[:, :-1], train_array[:, -1])

            model_path = os.path.join("Artifacts", "model.pkl")
            model = load_object(model_path)

            # Evaluate on test set
            y_test_pred = model.predict(X_test)
            test_rmse, test_mae, test_r2 = self.eval_metrics(y_test, y_test_pred)

            # Evaluate on train set
            y_train_pred = model.predict(X_train)
            train_rmse, train_mae, train_r2 = self.eval_metrics(y_train, y_train_pred)

            results = {
                "model": type(model).__name__,
                "train_metrics": {
                    "r2": round(float(train_r2), 4),
                    "mae": round(float(train_mae), 2),
                    "rmse": round(float(train_rmse), 2),
                },
                "test_metrics": {
                    "r2": round(float(test_r2), 4),
                    "mae": round(float(test_mae), 2),
                    "rmse": round(float(test_rmse), 2),
                },
            }

            # Save final evaluation results
            os.makedirs("Artifacts", exist_ok=True)
            eval_path = os.path.join("Artifacts", "final_model_metrics.json")
            with open(eval_path, 'w') as f:
                json.dump(results, f, indent=2)

            print("\n" + "=" * 60)
            print("FINAL MODEL EVALUATION")
            print("=" * 60)
            print(f"Model: {results['model']}")
            print(f"\nTrain Metrics:")
            print(f"  R²:   {train_r2:.4f}")
            print(f"  MAE:  {train_mae:.2f}")
            print(f"  RMSE: {train_rmse:.2f}")
            print(f"\nTest Metrics:")
            print(f"  R²:   {test_r2:.4f}")
            print(f"  MAE:  {test_mae:.2f}")
            print(f"  RMSE: {test_rmse:.2f}")
            print("=" * 60 + "\n")

            logger.info(f"Final model evaluation - Test R2: {test_r2:.4f}, MAE: {test_mae:.2f}, RMSE: {test_rmse:.2f}")
            logger.info(f"Evaluation results saved to {eval_path}")

            return results

        except Exception as e:
            raise e