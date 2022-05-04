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

data_root = '/home/ar88770/MarginLoss/promise_mc/valid'
model_path_ce = '/home/ar88770/MarginLoss/outputs/prostate_mc/unet-ce-adam/20220409-22:47:25-847975/best.pth'
model_path_ce_dice = '/home/ar88770/MarginLoss/outputs/prostate_mc/unet-ce_dice-adam/20220427-13:38:39-924288/best.pth'
model_path_focal = '/home/ar88770/MarginLoss/outputs/prostate_mc/unet-focal-adam/20220406-17:06:46-112063/best.pth'
model_path_penalty = '/home/ar88770/MarginLoss/outputs/prostate_mc/unet-penalty_ent-adam/20220411-23:10:20-338365/best.pth'
model_path_ls = '/home/ar88770/MarginLoss/outputs/prostate_mc/unet-ls-adam/20220409-22:57:05-534513/best.pth'
model_path_svls = '/home/ar88770/MarginLoss/outputs/prostate_mc/unet-svls-adam/20220417-14:10:04-419455/best.pth'
model_path_margin = '/home/ar88770/MarginLoss/outputs/prostate_mc/unet-logit_margin-adam/20220422-11:32:07-027131/best.pth'
in_channels = 1
nclasses = 3
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
            
        for sno in range(volimg.shape[0]):
            
            img = volimg[sno,:,:] 
            mask = volmask[sno,:,:]            
            
            image = img[80:240,80:240]
            mask = mask[80:240,80:240]
            
            imgT = torch.from_numpy(image)
            imgT = imgT.unsqueeze(0).unsqueeze(0)
            imgT = imgT.cuda('cuda:4')
                    
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