import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class ResnetDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        """
        Args:
            data_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data_dir = data_dir
        self.transform = transform
        # Example: Assuming that data_dir contains subdirectories for each class.
        # You can modify this part based on how your data is organized.
        self.classes = sorted(os.listdir(data_dir))
        self.images = []
        self.labels = []
        for label, cls in enumerate(self.classes):
            class_dir = os.path.join(data_dir, cls)
            # We assume each class directory contains image files.
            for fname in os.listdir(class_dir):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    self.images.append(os.path.join(class_dir, fname))
                    self.labels.append(label)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Load image and label
        img_path = self.images[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        # Optionally convert label to tensor if needed.
        # For classification problems in PyTorch, labels are often kept as integers.
        return image, label
