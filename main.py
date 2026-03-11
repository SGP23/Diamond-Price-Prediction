"""
Diamond Price Prediction System — Main Entry Point

Usage:
    python main.py              Run the training pipeline, then start the web server
    python main.py --train      Run only the training pipeline
    python main.py --serve      Start only the Flask web server
"""

import argparse
import os
import logging

# Initialize logging configuration
from src.DiamondPricePrediction.logger import logging as _  # noqa: F401

logger = logging.getLogger(__name__)


def train():
    """Run the full ML training pipeline."""
    from src.DiamondPricePrediction.pipelines.Training_pipeline import run_training_pipeline
    logger.info("Starting training pipeline...")
    run_training_pipeline()
    logger.info("Training pipeline completed.")


def serve():
    """Start the Flask prediction web server."""
    from app import app
    logger.info("Starting Flask server on http://localhost:8080")
    print("\n" + "=" * 50)
    print(" Diamond Price Prediction — Web Server")
    print(" Open http://localhost:8080 in your browser")
    print("=" * 50 + "\n")
    app.run(host="0.0.0.0", port=8080, debug=False)


def main():
    parser = argparse.ArgumentParser(description="Diamond Price Prediction System")
    parser.add_argument("--train", action="store_true", help="Run the training pipeline only")
    parser.add_argument("--serve", action="store_true", help="Start the web server only")
    args = parser.parse_args()

    if args.train:
        train()
    elif args.serve:
        # Check if model artifacts exist
        model_path = os.path.join("Artifacts", "model.pkl")
        if not os.path.exists(model_path):
            logger.warning("No trained model found. Running training pipeline first...")
            train()
        serve()
    else:
        # Default: train then serve
        model_path = os.path.join("Artifacts", "model.pkl")
        if not os.path.exists(model_path):
            print("No trained model found. Running training pipeline first...\n")
            train()
            print()
        serve()


if __name__ == "__main__":
    main()
