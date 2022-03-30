from torch.utils.data import Dataset, DataLoader
import torch


import h5py
import numpy as np
import os
import glob

CLASSES = ('background','foreground1','foreground2','foreground3','foreground4')

class BrainAtlasDataset(Dataset):
    def __init__(self, file_names):
        self.file_names = file_names
        self.classes = CLASSES
        self.info = []

        for fpath in self.file_names:
            with h5py.File(fpath, 'r') as hf:
                vol = hf['mask'][:]
            for ii in range(vol.shape[-1]):
                self.info.append([fpath,ii])

    def __len__(self):
        return len(self.info)

    def __getitem__(self,idx):
        img_file_name, sliceno  = self.info[idx]

        with h5py.File(img_file_name, 'r') as data:

            image = data["img"][:][:,:,:,sliceno]
            image = np.pad(image,pad_width=((0,0),(8,8),(8,8)),mode='constant')
            mask = data["mask"][:][:,:,sliceno]
            mask = np.pad(mask,pad_width=((8,8),(8,8)),mode='constant')

        return torch.from_numpy(image).float(), torch.from_numpy(mask).long()


def get_train_val_loader(data_root, batch_size=32, num_workers=8, pin_memory=True):

    train_path = os.path.join(data_root, 'train')
    train_files = glob.glob(train_path + '/*')

    valid_path = os.path.join(data_root, 'valid')
    valid_files = glob.glob(valid_path + '/*')

    train_dataset = BrainAtlasDataset(train_files)
    valid_dataset = BrainAtlasDataset(valid_files)

    display_dataset = [valid_dataset[i] for i in range(0, len(valid_dataset), len(valid_dataset) // 16)] # num.of.images for visualization 

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    display_loader = DataLoader(display_dataset, batch_size=8, drop_last=True)

    return train_loader, valid_loader, display_loader

def get_test_loader(data_root, batch_size=32, num_workers=8, pin_memory=True):

    test_path = os.path.join(data_root, 'valid')
    test_files = glob.glob(test_path + '/*')
    test_dataset = BrainAtlasDataset(test_files)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)

    return test_loader