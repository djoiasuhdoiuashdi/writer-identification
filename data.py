# python
import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets.vision import VisionDataset

class ResnetDataset(VisionDataset):
    def __init__(self, root: str, transform=None, target_transform=None, train: bool = True) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.train = train

        # Use sorted subfolder names as class names.
        self.classes = sorted(entry for entry in os.listdir(root)
                              if os.path.isdir(os.path.join(root, entry)))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        # Build a list of image file paths and targets (without preloading image data).
        self.samples = []
        for label, cls in enumerate(self.classes):
            class_dir = os.path.join(root, cls)
            for fname in os.listdir(class_dir):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    path = os.path.join(class_dir, fname)
                    self.samples.append((path, label))

    def __getitem__(self, index: int):
        # Retrieve file path and corresponding target.
        img_path, target = self.samples[index]

        # Load the image file on the fly.
        img = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self) -> int:
        return len(self.samples)