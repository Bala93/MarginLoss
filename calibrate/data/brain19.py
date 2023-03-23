from torch.utils.data import Dataset, DataLoader
import torch


import h5py
import numpy as np
import os
import glob
from sklearn.model_selection import train_test_split

CLASSES = ('background','foreground1','foreground2','foreground3')

class BrainDataset(Dataset):
    def __init__(self, file_names, mode='train'):
        self.file_names = file_names
        self.classes = CLASSES
        self.info = []
        self.mode = mode

        for fpath in self.file_names:
            
            if self.mode == 'train':
            
                with h5py.File(fpath, 'r') as hf:
                    vol = hf['mask'][:]
                for ii in range(vol.shape[1]):
                    self.info.append([fpath,ii])

            if self.mode == 'test':
                self.info.append([fpath, None])                    
            

    def __len__(self):
        return len(self.info)

    def __getitem__(self,idx):
        img_file_name, sliceno  = self.info[idx]
            
        with h5py.File(img_file_name, 'r') as data:

            volimg = data["img"][:]
            volmask = data["mask"][:]
            
        if self.mode == 'train':
            
            image = volimg[:,sliceno,:,:]
            mask = volmask[0,sliceno,:,:]
                
            return torch.from_numpy(image).float(), torch.from_numpy(mask).long()
        
        if self.mode == 'test':
            
            volimg = torch.from_numpy(volimg).permute(1,0,2,3)
            volmask = torch.from_numpy(volmask)
            
            return volimg.float(), volmask.long()             

def get_train_val_loader(data_root, batch_size=32, train_shuffle=True, valid_shuffle=False, num_workers=8, pin_memory=True):

    train_path = os.path.join(data_root, 'train')
    train_files = glob.glob(train_path + '/*')

    valid_path = os.path.join(data_root, 'valid')
    valid_files = glob.glob(valid_path + '/*')

    train_dataset = BrainDataset(train_files)
    valid_dataset = BrainDataset(valid_files)

    display_dataset = [valid_dataset[i] for i in range(0, len(valid_dataset), len(valid_dataset) // 16)] # num.of.images for visualization 

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=train_shuffle, num_workers=num_workers, pin_memory=pin_memory)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=valid_shuffle, num_workers=num_workers, pin_memory=pin_memory)

    display_loader = DataLoader(display_dataset, batch_size=8, drop_last=True)

    return train_loader, valid_loader, display_loader

def get_test_loader(data_root, batch_size=128, num_workers=8, pin_memory=True):

    test_path = os.path.join(data_root, 'test')
    test_files = glob.glob(test_path + '/*')
    test_dataset = BrainDataset(test_files, 'test')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    return test_loader

def get_post_temp_scaling_loader(data_root, batch_size=32, num_workers=8, pin_memory=True):

    valid_path = os.path.join(data_root, 'valid')
    test_path = os.path.join(data_root, 'test')
    
    valid_files = glob.glob(valid_path + '/*')
    test_files = glob.glob(test_path + '/*')
    
    total_files = valid_files + test_files
    
    train_files, test_files = train_test_split(total_files, test_size=0.2,random_state=42)
    train_files, valid_files = train_test_split(train_files, test_size=0.1,random_state=42)
    
    train_dataset = BrainDataset(train_files)
    valid_dataset = BrainDataset(valid_files)
    test_dataset = BrainDataset(test_files)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    return train_loader, valid_loader, test_loader
