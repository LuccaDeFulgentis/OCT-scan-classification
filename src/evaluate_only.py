from pathlib import Path
import logging

import torch

from src.model import OCTModel
from src.evaluate import evaluate_model


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)


MODEL_PATH = Path("saved_models/best_model.pth")
TEST_DIR = Path("data/OCT2017/test")

def main():
    """
    Loads a trained OCT model and evaluates it on the test dataset.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    model = OCTModel()

    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=device)
    )

    model.to(device)
    model.eval()

    logging.info("Evaluating saved model...")
    results = evaluate_model(model, TEST_DIR)
    logging.info(
        f"Final Test Accuracy: {results['accuracy']:.2f}%"
    )


if __name__ == "__main__":
    main()