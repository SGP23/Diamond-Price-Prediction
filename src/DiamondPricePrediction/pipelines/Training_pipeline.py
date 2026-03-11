import os
import logging
import numpy as np

from src.DiamondPricePrediction.components.Data_ingestion import DataIngestion
from src.DiamondPricePrediction.components.Data_transformation import DataTransformation
from src.DiamondPricePrediction.components.Model_trainer import ModelTrainer
from src.DiamondPricePrediction.components.Model_evaluation import ModelEvaluation

logger = logging.getLogger(__name__)


def generate_feature_importance(model, feature_names):
    """Generate and save feature importance visualization for the best model."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    os.makedirs("reports", exist_ok=True)

    # Get feature importances
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_)
    else:
        logger.info("Model does not support feature importance extraction.")
        return

    # Sort by importance
    indices = np.argsort(importances)[::-1]
    sorted_names = [feature_names[i] for i in indices]
    sorted_importances = importances[indices]

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(sorted_names)))
    bars = ax.barh(range(len(sorted_names)), sorted_importances, color=colors)
    ax.set_yticks(range(len(sorted_names)))
    ax.set_yticklabels(sorted_names, fontsize=11)
    ax.invert_yaxis()
    ax.set_xlabel('Feature Importance', fontsize=12)
    ax.set_title('Feature Importance — Best Model', fontsize=14, fontweight='bold')

    # Add value labels
    for bar, val in zip(bars, sorted_importances):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                f'{val:.4f}', va='center', fontsize=10)

    plt.tight_layout()
    save_path = os.path.join("reports", "feature_importance.png")
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Feature importance plot saved to {save_path}")
    print(f"\nFeature importance plot saved to: {save_path}")


def run_training_pipeline():
    """Execute the full training pipeline: ingestion -> transformation -> training -> evaluation."""
    logger.info("=" * 60)
    logger.info("TRAINING PIPELINE STARTED")
    logger.info("=" * 60)

    # Step 1: Data Ingestion
    logger.info("Step 1: Data Ingestion")
    obj = DataIngestion()
    train_data_path, test_data_path = obj.initiate_data_ingestion()
    logger.info(f"Data ingestion complete. Train: {train_data_path}, Test: {test_data_path}")

    # Step 2: Data Transformation
    logger.info("Step 2: Data Transformation")
    data_transformation = DataTransformation()
    train_arr, test_arr = data_transformation.initialize_data_transformation(train_data_path, test_data_path)
    logger.info(f"Data transformation complete. Train shape: {train_arr.shape}, Test shape: {test_arr.shape}")

    # Step 3: Model Training
    logger.info("Step 3: Model Training")
    model_trainer_obj = ModelTrainer()
    best_model_name, best_model = model_trainer_obj.initate_model_training(train_arr, test_arr)
    logger.info("Model training complete.")

    # Step 4: Model Evaluation
    logger.info("Step 4: Model Evaluation")
    model_eval_obj = ModelEvaluation()
    model_eval_obj.initiate_model_evaluation(train_arr, test_arr)
    logger.info("Model evaluation complete.")

    # Step 5: Feature Importance
    logger.info("Step 5: Feature Importance Visualization")
    feature_names = ['carat', 'depth', 'table', 'x', 'y', 'z', 'cut', 'color', 'clarity']
    generate_feature_importance(best_model, feature_names)

    logger.info("=" * 60)
    logger.info("TRAINING PIPELINE COMPLETED SUCCESSFULLY")
    logger.info("=" * 60)


if __name__ == "__main__":
    run_training_pipeline()