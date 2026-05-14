import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import logging

from .dataset import OCTDataset
from .model import OCTModel
from torchvision import transforms



logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)


def get_train_transform():
    """
    Creates the preprocessing and augmentation pipeline for training images.

    Returns:
        Training image transformations.
    """
    train_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    return train_transform


def get_val_transform():
    """
    Creates the preprocessing pipeline for validation images.

    Returns:
        Validation image transformations.
    """
    val_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
        ])
    return val_transform


def validate_model(model, val_loader, device):
    """
    Evaluates model accuracy on the validation dataset.

    Args:
        model: Trained PyTorch model.
        val_loader: Validation DataLoader.
        device: CPU or CUDA device.

    Returns:
        Validation accuracy percentage.
    """
    model.eval()
    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

        val_acc = 100 * val_correct / val_total
    return val_acc


def train_model(train_dir, val_dir, epochs=5, batch_size=32, lr=0.001, checkpoint_path="saved_models/best_model.pth"):
    """
    Trains the OCTModel on the training dataset.

    Args:
        train_dir (str): Path to the training dataset directory.
        val_dir (str): Path to the validation dataset directory.
        epochs (int): Number of training epochs.
        batch_size (int): Number of samples per batch.
        lr (float): Learning rate for the optimizer.
        checkpoint_path (str): Path used to save the best-performing model 

    Returns:
        The trained model instance
        List: Each epochs history of loss and validation accuracy
    """

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # Sets its to GPU instead of CPU. Does not work on local if cpu.
    logging.info(f"Using device: {device}")

    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    model = OCTModel().to(device)


    train_data = OCTDataset(train_dir, transform=get_train_transform())
    val_data = OCTDataset(val_dir, transform=get_val_transform())

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True) # Creates batches
    val_loader = DataLoader(val_data, batch_size=batch_size)

    criterion = nn.CrossEntropyLoss() # Creates the loss function
    optimizer = optim.Adam(model.parameters(), lr=lr) # Sets the adam optomiser learning rate

    best_val_acc = 0.0
    history = []

    for epoch in range(epochs): 
        model.train() # Sets to train mode
        running_loss = 0.0 # Loss tracker
        for inputs, labels in train_loader: # For each batch
            inputs, labels = inputs.to(device), labels.to(device) # Move input and labels to correct gpu.

            optimizer.zero_grad() # Clear the gradients to prevent accumulation (limited gpu memory)
            outputs = model(inputs) # Computes predictions
            loss = criterion(outputs, labels) # compares the predictions to truth values
            loss.backward() # propogates to compute loss graident 
            optimizer.step() # Updates the model
            running_loss += loss.item() #Batch loss

        epoch_loss = running_loss / len(train_loader)
        val_acc = validate_model(model, val_loader, device)

        history.append({
            "epoch": epoch + 1,
            "loss": epoch_loss,
            "val_accuracy": val_acc
        })

        logging.info(
            f"Epoch {epoch+1}/{epochs} | "
            f"Loss: {epoch_loss:.4f} | "
            f"Val Accuracy: {val_acc:.2f}%"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), checkpoint_path)
            logging.info(f"Saved new best model with val accuracy: {best_val_acc:.2f}%")

    return model, history
         