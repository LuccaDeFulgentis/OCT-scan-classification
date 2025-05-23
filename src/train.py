import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from .dataset import OCTDataset
from .model import OCTModel

def train_model(train_dir, val_dir, epochs=5, batch_size=32, lr=0.001):
    """
    Trains the OCTModel on the training dataset.

    Args:
        train_dir (str): Path to the training dataset directory.
        val_dir (str): Path to the validation dataset directory.
        epochs (int): Number of training epochs.
        batch_size (int): Number of samples per batch.
        lr (float): Learning rate for the optimizer.

    Returns:
        The trained model instance.
    """

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # Sets its to GPU instead of CPU. Does not work on local if cpu.
    model = OCTModel().to(device)

    train_data = OCTDataset(train_dir)
    val_data = OCTDataset(val_dir)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True) # Creates batches
    val_loader = DataLoader(val_data, batch_size=batch_size)

    criterion = nn.CrossEntropyLoss() # Creates the loss function
    optimizer = optim.Adam(model.parameters(), lr=lr) # Sets the adam optomiser learning rate

    for epoch in range(epochs): # For each epoch
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

        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}")
    
    return model
