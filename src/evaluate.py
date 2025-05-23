import torch
from torch.utils.data import DataLoader
from .dataset import OCTDataset

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
    model.eval()  # Sets to evaluation mode

    test_data = OCTDataset(test_dir) #
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False) # Wraps the dataset in a dataloader

    correct = 0 
    total = 0

    with torch.no_grad(): # No graident calculations
        for inputs, labels in test_loader: # for each batch, inputs = tensors, labels are the true labels.
            inputs = inputs.to(device)     
            labels = labels.to(device)     

            outputs = model(inputs) # Output predictions
            _, predicted = torch.max(outputs.data, 1) # Predicted is the class index
            total += labels.size(0) #adds to the total number of samples
            correct += (predicted == labels).sum().item() # Compares the prediction to the label

    print(f"Accuracy on test set: {100 * correct / total:.2f}%")
