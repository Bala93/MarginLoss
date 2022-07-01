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

data_root = '/home/ar88770/MarginLoss/promise_mc/test'
model_path_ce = '/home/ar88770/MarginLoss/outputs/prostate_mc/unet-ce-adam/20220618-19:45:47-605216/best.pth'
model_path_ce_dice = '/home/ar88770/MarginLoss/outputs/prostate_mc/unet-ce_dice-adam/20220618-19:59:27-919203/best.pth'
model_path_focal = '/home/ar88770/MarginLoss/outputs/prostate_mc/unet-focal-adam/20220619-19:27:36-214693/best.pth'
model_path_penalty = '/home/ar88770/MarginLoss/outputs/prostate_mc/unet-penalty_ent-adam/20220618-20:26:38-657056/best.pth'
model_path_ls = '/home/ar88770/MarginLoss/outputs/prostate_mc/unet-ls-adam/20220619-18:30:02-915430/best.pth'
model_path_svls = '/home/ar88770/MarginLoss/outputs/prostate_mc/unet-svls-adam/20220618-20:55:27-830091/best.pth'
model_path_margin = '/home/ar88770/MarginLoss/outputs/prostate_mc/unet-logit_margin-adam/20220618-21:09:31-193817/best.pth'

in_channels = 1
nclasses = 3
method_names =['ce','ce_dice','penalty','focal','ls','svls','margin']

files = glob.glob('{}/*.h5'.format(data_root))
models_path = [model_path_ce, model_path_ce_dice, model_path_focal, model_path_penalty, model_path_ls, model_path_svls, model_path_margin]
model = UNet(input_channels=in_channels, num_classes=nclasses)

for key, model_path in zip(method_names,models_path):

    checkpoint = torch.load(model_path)["state_dict"]
    checkpoint = dict((key[7:] if "module" in key else key, value)for (key, value) in checkpoint.items())
    model.load_state_dict(checkpoint)
    model = model.to('cuda:4')
    
    metrics_dict = {"fname":[],"dsc":[],"hd":[],"ece":[],"cece":[]}
    savedir = os.path.dirname(model_path)
    metricspath = os.path.join(savedir, 'metrics3d.csv')

    for fpath in tqdm(files):

        fname = os.path.basename(fpath)
        
        with h5py.File(fpath, 'r') as hf:
            
            volimg = hf['img'][:]
            volmask = hf['mask'][:]
                        
            img = volimg[:,80:240,80:240]
            mask = volmask[:,80:240,80:240]
            
        imgT = torch.from_numpy(img)
        imgT = imgT.unsqueeze(1)
        imgT = imgT.cuda('cuda:4')
                
        target = np.expand_dims(mask,axis=1)
        targetT = torch.from_numpy(target[:,0])
        targetT = targetT.cuda('cuda:4')
        
        predT = model(imgT.float())
        
        outputconf = F.softmax(predT,dim=1).detach().cpu().numpy()
        output = np.argmax(outputconf,axis=1)
        dsc, hd = shape_metrics(output, mask, nclasses)

        n, c, x, y = predT.shape
        logits = torch.einsum("ncxy->nxyc", predT)
        logits = logits.reshape(n * x * y, -1)
        labels = targetT.reshape(n * x * y)
        
        index = torch.nonzero(labels != 0).squeeze()
        logits = logits[index, :]
        labels = labels[index]

        ece, cece = calib_metrics(logits, labels)
    
        metrics_dict['fname'].append(fname)
        metrics_dict['dsc'].append(dsc)
        metrics_dict['hd'].append(hd)
        metrics_dict['ece'].append(ece)
        metrics_dict['cece'].append(cece)
        
    df_metrics = pd.DataFrame(metrics_dict)
    df_metrics.loc['mean'] = df_metrics.mean().round(4)
    df_metrics.to_csv(metricspath)
