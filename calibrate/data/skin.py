from torch.utils.data import Dataset, DataLoader
import torch
from skimage.transform import resize

import h5py
import numpy as np
import os
import glob

CLASSES = ('background','foreground1')

class SkinDataset(Dataset):
    def __init__(self, file_names):
        self.file_names = file_names
        self.classes = CLASSES
        self.info = []

        for fpath in self.file_names:
            self.info.append(fpath)

    def __len__(self):
        return len(self.info)

    def __getitem__(self,idx):
        img_file_name = self.info[idx]

        with h5py.File(img_file_name, 'r') as data:

            image = data["img"][:]
            mask = data["mask"][:]
              
            mask = mask.astype(np.uint8)         

            image = np.expand_dims(image,axis=0)
            image = np.transpose(image, axes=[2,0,1])

        return torch.from_numpy(image).float(), torch.from_numpy(mask).long()


def get_train_val_loader(data_root, batch_size=32, num_workers=8, pin_memory=True):

    train_path = os.path.join(data_root, 'train')
    train_files = glob.glob(train_path + '/*')

    valid_path = os.path.join(data_root, 'valid')
    valid_files = glob.glob(valid_path + '/*')

    train_dataset = SkinDataset(train_files)
    valid_dataset = SkinDataset(valid_files)

    display_dataset = [valid_dataset[i] for i in range(0, len(valid_dataset), len(valid_dataset) // 16)] # num.of.images for visualization 

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    display_loader = DataLoader(display_dataset, batch_size=8, drop_last=True)

    return train_loader, valid_loader, display_loader

def get_test_loader(data_root, batch_size=32, num_workers=8, pin_memory=True):

    test_path = os.path.join(data_root, 'valid')
    test_files = glob.glob(test_path + '/*')
    test_dataset = SkinDataset(test_files)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)

    return test_loader