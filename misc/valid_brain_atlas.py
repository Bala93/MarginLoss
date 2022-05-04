import glob
from calibrate.net.unet import UNet
import numpy as np
import torch
from tqdm import tqdm
import h5py
from torch.nn import functional as F
import matplotlib.pyplot as plt
from skimage.transform import resize
import os
import pandas as pd
from utils import *

data_root = '/home/ar88770/MarginLoss/mrbrains/valid'
model_path_ce = '/home/ar88770/MarginLoss/outputs/brainatlas/unet-ce-adam/20220404-16:20:24-829672/best.pth'
model_path_ce_dice = '/home/ar88770/MarginLoss/outputs/brainatlas/unet-ce_dice-adam/20220427-17:25:09-405389/best.pth'
model_path_focal = '/home/ar88770/MarginLoss/outputs/brainatlas/unet-focal-adam/20220406-10:59:04-103885/best.pth'
model_path_penalty = '/home/ar88770/MarginLoss/outputs/brainatlas/unet-penalty_ent-adam/20220406-11:26:33-769991/best.pth'
model_path_ls = '/home/ar88770/MarginLoss/outputs/brainatlas/unet-ls-adam/20220404-16:49:47-579491/best.pth'
model_path_svls = '/home/ar88770/MarginLoss/outputs/brainatlas/unet-svls-adam/20220417-17:23:24-173305/best.pth'
model_path_margin = '/home/ar88770/MarginLoss/outputs/brainatlas/unet-logit_margin-adam/20220404-22:15:12-835794/best.pth'
in_channels = 3
nclasses = 4
method_names =['ce','ce_dice','penalty','focal','ls','svls','margin']

files = glob.glob('{}/*.h5'.format(data_root))
models_path = [model_path_ce, model_path_penalty, model_path_focal, model_path_ls, model_path_margin]
model = UNet(input_channels=in_channels, num_classes=nclasses)

for key, model_path in zip(method_names,models_path):

    checkpoint = torch.load(model_path)["state_dict"]
    checkpoint = dict((key[7:] if "module" in key else key, value)for (key, value) in checkpoint.items())
    model.load_state_dict(checkpoint)
    model = model.to('cuda:4')
    
    metrics_dict = {"fname":[],"sno":[], "dsc":[],"hd":[],"ece":[],"cece":[]}
    savedir = os.path.dirname(model_path)
    metricspath = os.path.join(savedir, 'metrics.csv')

    for fpath in tqdm(files):

        fname = os.path.basename(fpath)
        
        with h5py.File(fpath, 'r') as hf:
            volimg = hf['img'][:]
            volmask = hf['mask'][:]
            
        for sno in range(volimg.shape[-1]):
            
            img = volimg[:,:,:,sno] 
            img = np.pad(img,pad_width=((0,0),(8,8),(8,8)),mode='constant')
            imgT = torch.from_numpy(img)
            imgT = imgT.unsqueeze(0)
            imgT = imgT.cuda('cuda:4')
            
            mask = volmask[:,:,sno]            
            mask = np.pad(mask,pad_width=((8,8),(8,8)),mode='constant')
            mask[mask==4] = 0 # to remove fg4.

            target = np.expand_dims(mask,axis=0)
            targetT = torch.from_numpy(target)

            predT = model(imgT.float()).detach().cpu()
            ece, cece = calib_metrics(predT, targetT)
            
            outputconf = F.softmax(predT,dim=1).numpy()
            output = np.argmax(outputconf[0],axis=0)
            dsc, hd = shape_metrics(output, mask, nclasses)
        
            metrics_dict['fname'].append(fname)
            metrics_dict['sno'].append(sno)
            metrics_dict['dsc'].append(dsc)
            metrics_dict['hd'].append(hd)
            metrics_dict['ece'].append(ece)
            metrics_dict['cece'].append(cece)
        
    df_metrics = pd.DataFrame(metrics_dict)
    df_metrics.to_csv(metricspath)