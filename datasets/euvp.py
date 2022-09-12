import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import Dataset
from utils.data_utils import get_file_paths
from PIL import Image


class EUVPDataset(Dataset):
    def __init__(self, root, split='train', paired=True, transforms_=None):
        self.transform = transforms.Compose(transforms_)
        self.split = split
        self.paired = paired
        if self.split == 'train':
            self.train_a, self.train_b = get_file_paths(root=root, split=self.split, paired=self.paired)
            self.len = min(len(self.train_a), len(self.train_b))
        else:
            self.val = get_file_paths(root=root, split=self.split, paired=self.paired)
            self.len = len(self.val)

    def __getitem__(self, index):
        if self.split == 'train':
            img_a = Image.fromarray(np.array(Image.open(self.train_a[index % self.len]))[:, ::-1, :], "RGB")
            img_b = Image.fromarray(np.array(Image.open(self.train_b[index % self.len]))[:, ::-1, :], "RGB")
            img_a = self.transform(img_a)
            img_b = self.transform(img_b)
            return {"A": img_a, "B": img_b}
        else:
            img_val = Image.open(self.val[index % self.len])
            img_val = self.transform(img_val)
            return {"val": img_val}

    def __len__(self):
        return self.len
