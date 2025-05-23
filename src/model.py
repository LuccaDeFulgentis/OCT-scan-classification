import torch.nn as nn # Neural networks and layers
import torch.nn.functional as F # Activation functions such as pooling  

class OCTModel(nn.Module):
    """
    
    """
    def __init__(self, num_classes=4):
        """
        Initializes the CNN layers and parameters.
        
        Args:
            num_classes (int): Number of classes ie OCT classifications.

        Attributes:
            conv1: First convolutional layer (1 input channel, 32 output channels).
            pool : Max pooling layer to downsample.
            conv2 : Second convolutional layer (32 input channels, 64 output channels).
            fc1 : Fully connected layer that reduces flattened features to 128 units.
            dropout : Dropout layer with 50% dropping to prevent overfitting.
            fc2 : fully connected layer map
        """

        super(OCTModel, self).__init__() # Utalizes the nn.Module constructor 

        self.conv1 = nn.Conv2d(1, 32, 3, padding=1) # 1 input channel (Greyscale), 32 output channels, 3x3 kernal size, padding = 1 keeps same size.
        self.pool = nn.MaxPool2d(2, 2) # Pooling layer. Reduces the dimensions to 64x64 2x2 window
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1) # Second layer outputs 64 feature map
        self.fc1 = nn.Linear(64 * 32 * 32, 128) # Fully connected layer. 64 channels, 32 by 32
        self.dropout = nn.Dropout(0.5) # zeros 50% of elements. Prevents overfitting
        self.fc2 = nn.Linear(128, num_classes) # Final connected layer. 128 features.

    def forward(self, x):
        """
        Defines the forward pass of the model.
        
        Args:
            x : Input tensor with shape dimensions

        Returns:
            tensor: output logits with shape (batch_size, num_classes),
        """ 
        
        x = self.pool(F.relu(self.conv1(x)))  # Pools layer after relu (128x128 -> 64x64) Sets all negative values to 0s
        x = self.pool(F.relu(self.conv2(x)))  # Pools layer after relu (64x64 -> 32x32) Sets all negative values to 0s 
        x = x.view(-1, 64 * 32 * 32) # Flatten the feature maps
        x = F.relu(self.fc1(x)) # Fully connected and sets all negative values to 0
        x = self.dropout(x) # Dropout regularization
        x = self.fc2(x) # Final output layer for each class
        return x # returns the output logits
