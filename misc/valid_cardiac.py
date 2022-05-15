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

data_root = '/home/ar88770/MarginLoss/acdc/valid'
model_path_ce = '/home/ar88770/MarginLoss/outputs/cardiac/unet-ce-adam/20220323-20:17:06-951406/best.pth'
model_path_ce_dice = '/home/ar88770/MarginLoss/outputs/cardiac/unet-ce_dice-adam/20220427-13:35:35-177337/best.pth'
model_path_focal = '/home/ar88770/MarginLoss/outputs/cardiac/unet-focal-adam/20220323-21:00:04-839009/best.pth'
model_path_penalty = '/home/ar88770/MarginLoss/outputs/cardiac/unet-penalty_ent-adam/20220323-22:46:02-093977/best.pth'
model_path_ls = '/home/ar88770/MarginLoss/outputs/cardiac/unet-ls-adam/20220324-17:38:33-210962/best.pth'
model_path_svls = '/home/ar88770/MarginLoss/outputs/cardiac/unet-svls-adam/20220417-12:03:39-233516/best.pth'
model_path_margin = '/home/ar88770/MarginLoss/outputs/cardiac/unet-logit_margin-adam/20220324-14:58:28-642625/best.pth'
in_channels = 1
nclasses = 4
method_names =['ce','ce_dice','penalty','focal','ls','svls','margin']

files = glob.glob('{}/*.h5'.format(data_root))
models_path = [model_path_ce, model_path_ce_dice, model_path_focal, model_path_penalty, model_path_ls, model_path_svls, model_path_margin]
model = UNet(input_channels=in_channels, num_classes=nclasses)

for key, model_path in zip(method_names,models_path):

    checkpoint = torch.load(model_path,map_location=torch.device('cpu'))["state_dict"]
    checkpoint = dict((key[7:] if "module" in key else key, value)for (key, value) in checkpoint.items())
    model.load_state_dict(checkpoint)
    model = model.to('cuda:5')
    
    metrics_dict = {"fname":[],"dsc":[],"hd":[],"ece":[],"cece":[]}
    savedir = os.path.dirname(model_path)
    metricspath = os.path.join(savedir, 'metrics.csv')
    resultsdir = os.path.join(savedir, 'results')
    
    if not os.path.exists(resultsdir):
        os.mkdir(resultsdir)

    for fpath in tqdm(files):

        fname = os.path.basename(fpath)
        resultspath = os.path.join(resultsdir, fname)
        
        with h5py.File(fpath, 'r') as hf:
            img = hf['img'][:]
            mask = hf['mask'][:]
            
        img = np.pad(img,pad_width=((5,5),(5,5)),mode='constant')
        mask = np.pad(mask,pad_width=((5,5),(5,5)),mode='constant')

        imgT = torch.from_numpy(img)
        imgT = imgT.unsqueeze(0).unsqueeze(0)
        imgT = imgT.cuda('cuda:5')

        target = np.expand_dims(mask,axis=0)
        targetT = torch.from_numpy(target)

        predT = model(imgT.float()).detach().cpu()
        ece, cece = calib_metrics(predT, targetT)
        
        outputconf = F.softmax(predT,dim=1).numpy()
        output = np.argmax(outputconf[0],axis=0)
        
        with h5py.File(resultspath, 'w') as hf:
            hf['mask'] = output
                    
        dsc, hd = shape_metrics(output, mask, nclasses)
    
        metrics_dict['fname'].append(fname)
        metrics_dict['dsc'].append(dsc)
        metrics_dict['hd'].append(hd)
        metrics_dict['ece'].append(ece)
        metrics_dict['cece'].append(cece)
        
    df_metrics = pd.DataFrame(metrics_dict)
    df_metrics.to_csv(metricspath)