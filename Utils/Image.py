from torch.utils.data import Dataset

class OriginalDataset(Dataset):
    def __init__(self, imgs, labels, transform=None):
        self.imgs = imgs
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        imgs = self.imgs[idx]
        labels = self.labels[idx]
        if self.transform:
            imgs = self.transform(imgs)
        return imgs, labels