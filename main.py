import os
import torch

from src.train import train_model
from src.evaluate import evaluate_model

# Paths to dataset directories
BASE_DIR = "/home/ludef/OCT_scan_classification/data/OCT2017"
TRAIN_DIR = os.path.join(BASE_DIR, "train")
VAL_DIR = os.path.join(BASE_DIR, "val")
TEST_DIR = os.path.join(BASE_DIR, "test")

def main():
    print("Training started")  

    # Train the model
    model = train_model(
        train_dir=TRAIN_DIR,
        val_dir=VAL_DIR,
        epochs=5,
        batch_size=32,
        lr=0.001
    )

    # Save model
    model_path = "oct_model.pth"
    torch.save(model.state_dict(), model_path)
    print(f"\nModel saved to {model_path}")

    # Evaluate model
    print("Evaluating model")
    evaluate_model(model, TEST_DIR)

if __name__ == "__main__":
    main()
