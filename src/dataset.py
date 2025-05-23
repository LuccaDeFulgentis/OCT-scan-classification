import os
from PIL import Image # Image loader
from torch.utils.data import Dataset # How to load and access data
import torchvision.transforms as transforms # Pre-Processing tools 

# Class that loads and processes the OCT images
class OCTDataset(Dataset): 
    """
    A dataset for loading and preprocessing images organized in subdirectories.
    The images are loaded, converted to grayscale, resized to 128x128 pixels, and transformed into PyTorch tensors.

    Args:
        root_dir (str): Path to the root directory containing subdirectories for each class. 

    Attributes:
        root_dir (str): Root directory path.
        class_to_idx (dict): Mapping from class folder names to integer labels.
        samples (list): List of (image_path, label) for all images in the dataset.
        transform (torch function): Image transformations applied to each image.

    Methods:
        __len__(): Returns the total number of images in the dataset.
        __getitem__(idx): Loads and returns the transformed image tensor and its label at the given index.
    """

    def __init__(self, root_dir): 
        """
        Initializes the OCTDataset.

        Args:
            root_dir (str): 
                Path to the root directory containing subdirectories for each class.

        Attributes:
            class_to_idx (dict): Dictionary mapping class folder names to integer labels.
            samples (list): List of (image_path, label) for loading data during training.
    """

        self.root_dir = root_dir # New objects root directory set to given arg 
        self.transform = transforms.Compose([  # Transforms the images .compose allows to chain the transformations
            transforms.Grayscale(),  # Transforms the image to grayscale (should be spelled greyscale!)
            transforms.Resize((128, 128)), # Resizes the images to 128 by 128
            transforms.ToTensor(), # Converts the image into a PyTorch Tensor (C * H * W) 
            # (Channels, Height, Width) then each pixel will have a value [0, 1.0] for the intensity 
        ])

        self.samples = []
        self.class_to_idx = {class_name: idx for idx, class_name in enumerate(os.listdir(root_dir))} # Creates a dictionary with each subfolder and index ie [class: idx,]
        
        for label in os.listdir(root_dir): # For each subfolders label = subfolder
            class_dir = os.path.join(root_dir, label) # Enter each subfolder
            for img_name in os.listdir(class_dir): # For each image in the current subfolder
                img_path = os.path.join(class_dir, img_name) # Update image path
                self.samples.append((img_path, self.class_to_idx[label])) # List [(image path : subfolder index)] [('OCT2017/CNV/img1.jpeg', 0), ('OCT2017/CNV/img2.jpeg', 0)]

    def __len__(self):
        """
        Returns the total number of samples in the dataset.

        Returns:
            int: The number of image samples in the dataset.
        """
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Retrieves the image and label at the specified index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: (image, label) where
                - image (Tensor): The transformed image tensor.
                - label (int): The integer label corresponding to the image class.

        """
        img_path, label = self.samples[idx] 
        image = Image.open(img_path)
        image = self.transform(image) # Converts image into a tensor

        return image, label
