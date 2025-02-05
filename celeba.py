import torch
from torch.utils.data import Dataset
from torchvision.datasets.celeba import CelebA
import torchvision.transforms.functional as F
from torchvision import transforms

# the cropping did in original DiTi paper. 
# Aggressive but focus on face and eliminates background noise
class CropCelebA(object):
    def __call__(self, img):
        new_img = F.crop(img, 57, 25, 128, 128)
        return new_img 

class MyCelebA(Dataset):
    def __init__(self, root, phase):
        assert phase in ['train', 'valid', 'test']
        self.celeba_ds = self._load_dataset(root, phase)
        self.img_transform = transforms.Compose([CropCelebA(), 
                                                 transforms.ToTensor()])
                                                 
        
    def _load_dataset(self, root, phase):
        return CelebA(root,
                      split=phase,
                      download=True)

    def __getitem__(self, index):
        img, _ = self.celeba_ds[index]
        return self.img_transform(img)

    def __len__(self):
        return len(self.celeba_ds)
