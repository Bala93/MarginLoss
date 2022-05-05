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

data_root = '/home/ar88770/MarginLoss/refuge/valid'
model_path_ce = '/home/ar88770/MarginLoss/outputs/fundus/unet-ce-adam/20220401-16:19:09-606441/best.pth'
model_path_ce_dice = '/home/ar88770/MarginLoss/outputs/fundus/unet-ce_dice-adam/20220427-11:21:48-510735/best.pth'
model_path_focal = '/home/ar88770/MarginLoss/outputs/fundus/unet-focal-adam/20220406-16:17:33-121435/best.pth'
model_path_penalty = '/home/ar88770/MarginLoss/outputs/fundus/unet-penalty_ent-adam/20220406-17:10:29-709839/best.pth'
model_path_ls = '/home/ar88770/MarginLoss/outputs/fundus/unet-ls-adam/20220401-22:06:25-082822/best.pth'
model_path_svls = '/home/ar88770/MarginLoss/outputs/fundus/unet-svls-adam/220220417-14:08:54-166306/best.pth'
model_path_margin = '/home/ar88770/MarginLoss/outputs/fundus/unet-logit_margin-adam/20220406-15:20:13-315048/best.pth'
in_channels = 3
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
    
    metrics_dict = {"fname":[],"dsc":[],"hd":[],"ece":[],"cece":[]}
    savedir = os.path.dirname(model_path)
    metricspath = os.path.join(savedir, 'metrics.csv')

    for fpath in tqdm(files):

        fname = os.path.basename(fpath)
        
        
        with h5py.File(fpath, 'r') as data:

            image = data["img"][:]
            mask = data["mask"][:]

            mask[mask==0] = 2
            mask[mask==128] = 1
            mask[mask==255] = 0

            image = resize(image,[256,256],order=3,preserve_range=True)
            mask = resize(mask, [256,256],order=0,preserve_range=True)               
            img = np.transpose(image, axes=[2,0,1])

        imgT = torch.from_numpy(img)
        imgT = imgT.unsqueeze(0)
        imgT = imgT.cuda('cuda:4')

        target = np.expand_dims(mask,axis=0)
        targetT = torch.from_numpy(target)

        predT = model(imgT.float()).detach().cpu()
        ece, cece = calib_metrics(predT, targetT)
        
        outputconf = F.softmax(predT,dim=1).numpy()
        output = np.argmax(outputconf[0],axis=0)
        dsc, hd = shape_metrics(output, mask, nclasses)
    
        metrics_dict['fname'].append(fname)
        metrics_dict['dsc'].append(dsc)
        metrics_dict['hd'].append(hd)
        metrics_dict['ece'].append(ece)
        metrics_dict['cece'].append(cece)
        
    df_metrics = pd.DataFrame(metrics_dict)
    df_metrics.to_csv(metricspath)