import torch.nn as nn 
from torchvision import models

class OCTModel(nn.Module):
    """
        CNN Model for OCT Classification
    """
    def __init__(self, num_classes=4):
        """
        Initializes the pretrained ResNet-18 model and replaces
        the final classification layer.

        Args:
            num_classes (int): Number of OCT output classes.
        """

        super(OCTModel, self).__init__() # Utalizes the nn.Module constructor 

        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT) #Pretrained ResNet-18 
 
        in_features = self.model.fc.in_features #Number of features 

        self.model.fc = nn.Sequential( #replace the final layer
            nn.Dropout(0.5), 
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        """
        Defines the forward pass of the model.

        Args:
            x (Tensor): Input image batch.

        Returns:
            Tensor: Output logits for each class.
        """
        return self.model(x)
