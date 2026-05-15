import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

from pathlib import Path
import logging


from .dataset import OCTDataset

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def get_test_transform():
    """
    Creates the preprocessing pipeline for test images.

    Returns:
        Test image transformations.
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

def save_confusion_matrix(cm, class_names):
    """
    Saves a confusion matrix plot

    Args:
        cm: Confusion matrix values.
        class_names: List of class names.
    """

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.colorbar()

    tick_marks = range(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            plt.text(j, i, cm[i, j], ha="center", va="center")

    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    plt.close()


def evaluate_model(model, test_dir, batch_size=32):
    """
    Evaluates the model's performance on the test dataset.

    Args:
        model (pytorch object): The trained PyTorch model to evaluate.
        test_dir (str): Path to the test dataset directory.
        batch_size (int): Number of samples per batch set to 32.

    Returns:
        N/A
        Prints the accuracy of the model on the test dataset.

    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Sets its to GPU instead of CPU. Does not work on local if cpu.
    model.to(device)
    model.eval() 

    test_data = OCTDataset(test_dir, transform=get_test_transform())
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=2) # Wraps the dataset in a dataloader

    all_preds = []
    all_labels = []

    with torch.no_grad(): # No graident calculations
        for inputs, labels in test_loader: # for each batch, inputs = tensors, labels are the true labels.
            inputs = inputs.to(device)     
            labels = labels.to(device)     

            outputs = model(inputs) # Output predictions
            _, predicted = torch.max(outputs, 1) # Predicted is the class index

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    class_names = test_data.classes

    accuracy = accuracy_score(all_labels, all_preds) * 100

    report = (classification_report(all_labels, all_preds, target_names=class_names))

    cm = confusion_matrix(all_labels, all_preds)

    logging.info(f"Test Accuracy: {accuracy:.2f}%")
    logging.info("\nClassification Report:\n%s", report)
    logging.info("\nConfusion Matrix:\n%s", cm)

    save_confusion_matrix(cm, class_names)

    return {
        "accuracy": accuracy,
        "classification_report": report,
        "confusion_matrix": cm
    }