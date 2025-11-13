from torch.utils.data import Dataset

class OriginalDataset(Dataset):
    def __init__(self, args, imgs, imgs_other, labels, transform=None):
        self.args = args
        self.imgs = imgs
        self.imgs_other = imgs_other
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        labels = self.labels[idx]
        if self.args.view in ['Top', 'Side']:
            imgs = self.imgs[idx]
            if self.transform:
                imgs = self.transform(imgs)
            return imgs, labels
        elif self.args.view == 'Fusion':
            imgs = self.imgs[idx]
            imgs_other = self.imgs_other[idx]
            if self.transform:
                imgs = self.transform(imgs)
                imgs_other = self.transform(imgs_other)
            return (imgs, imgs_other), labels
        else:
            raise ValueError(f'Invalid view type: {self.args.view}')