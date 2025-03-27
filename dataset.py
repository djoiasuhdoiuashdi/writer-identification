import os
from PIL import Image
from torchvision.datasets.vision import VisionDataset

class ResnetDataset(VisionDataset):
    def __init__(self, root: str) -> None:
        super().__init__(root)
        self.classes = sorted(entry for entry in os.listdir(root))
        self.samples = []
        for label, cls in enumerate(self.classes):
            class_dir = os.path.join(root, cls)
            for fname in os.listdir(class_dir):
                path = os.path.join(class_dir, fname)
                self.samples.append((path, label))

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        img = Image.open(path).convert("1")
        return img, label

    def __len__(self) -> int:
        return len(self.samples)