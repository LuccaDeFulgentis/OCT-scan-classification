import os
from PIL import Image # Image loader
from torch.utils.data import Dataset # How to load and access data


# Class that loads and processes the OCT images
class OCTDataset(Dataset): 
    """
    PyTorch dataset for OCT retinal image classification.

    Loads images from class-specific subdirectories and applies optional
    image transformations for training and evaluation

    """

    def __init__(self, root_dir, transform=None): 
        """
        Initializes the OCT dataset.

        Args:
            root_dir: Path to the dataset directory.
            transform: Optional image transformations.
        """

        self.root_dir = root_dir # New objects root directory set to given arg 
        self.transform = transform

        self.classes = sorted([
            class_name for class_name in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, class_name))
        ])

        self.class_to_idx = {
            class_name: idx for idx, class_name in enumerate(self.classes)
        }

        self.samples = []
        
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            label = self.class_to_idx[class_name]

            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith((".png", ".jpg", ".jpeg")):
                    img_path = os.path.join(class_dir, img_name)
                    self.samples.append((img_path, label))

        
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
        image = Image.open(img_path).convert("RGB")
        if self.transform :
            image = self.transform(image) # Converts image into a tensor

        return image, label
