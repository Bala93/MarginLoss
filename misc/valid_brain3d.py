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


data_root = '/home/ar88770/MarginLoss/brats/test'
model_path_ce = '/home/ar88770/MarginLoss/outputs/brain/unet-ce-adam/20220706-22:15:12-802191/best.pth'
model_path_ce_dice = '/home/ar88770/MarginLoss/outputs/brain/unet-ce_dice-adam/20220706-22:43:45-223444/best.pth'
model_path_focal = '/home/ar88770/MarginLoss/outputs/brain/unet-focal-adam/20220706-23:11:36-654202/best.pth'
model_path_penalty = '/home/ar88770/MarginLoss/outputs/brain/unet-penalty_ent-adam/20220707-06:30:38-357512/best.pth'
model_path_ls = '/home/ar88770/MarginLoss/outputs/brain/unet-ls-adam/20220707-04:39:23-038720/best.pth'
model_path_svls = '/home/ar88770/MarginLoss/outputs/brain/unet-svls-adam/20220707-00:40:49-177623/best.pth'
model_path_margin = '/home/ar88770/MarginLoss/outputs/brain/unet-logit_margin-adam/20220707-02:20:08-331134/best.pth'

in_channels = 4
nclasses = 4
method_names =['ce','ce_dice','penalty','focal','ls','svls','margin']

files = glob.glob('{}/*.h5'.format(data_root))
models_path = [model_path_ce, model_path_ce_dice, model_path_focal, model_path_penalty, model_path_ls, model_path_svls, model_path_margin]
model = UNet(input_channels=in_channels, num_classes=nclasses)
gpuid = 5


for key, model_path in zip(method_names,models_path):

    checkpoint = torch.load(model_path,map_location=torch.device('cpu'))["state_dict"]
    checkpoint = dict((key[7:] if "module" in key else key, value)for (key, value) in checkpoint.items())
    model.load_state_dict(checkpoint)
    model = model.to('cuda:{}'.format(gpuid))
    
    metrics_dict = {"fname":[],"dsc":[],"hd":[],"ece":[],"cece":[]}
    savedir = os.path.dirname(model_path)
    metricspath = os.path.join(savedir, 'metrics3d.csv')

    resultsdir = os.path.join(savedir, 'results')

    if not os.path.exists(resultsdir):
        os.mkdir(resultsdir)

    for fpath in tqdm(files):

        fname = os.path.basename(fpath)

        resultspath = os.path.join(resultsdir, fname)
        
        with h5py.File(fpath, 'r') as hf:
            img = hf['img'][:]
            target = hf['mask'][:]

        img = img.transpose(2,3,0,1)
        target = target.transpose(2,0,1)
        
        imgT = torch.from_numpy(img)
        imgT = imgT.cuda('cuda:{}'.format(gpuid))
        
        targetT = torch.from_numpy(target)

        predT = model(imgT.float()).detach().cpu()
        
        outputconf = F.softmax(predT,dim=1).numpy()
        output = np.argmax(outputconf,axis=1)
    
        #output, target = bratspostprocess(output, target)
        
        
        dsc, hd = shape_metrics(output, target, nclasses)

        n, c, x, y = predT.shape
        logits = torch.einsum("ncxy->nxyc", predT)
        logits = logits.reshape(n * x * y, -1)
        labels = targetT.reshape(n * x * y)
        
        index = torch.nonzero(labels != 0).squeeze()
        logits = logits[index, :]
        labels = labels[index]

        #ece, cece = calib_metrics(predT, targetT)
        ece, cece = calib_metrics(logits, labels)
    
        metrics_dict['fname'].append(fname)
        metrics_dict['dsc'].append(dsc)
        metrics_dict['hd'].append(hd)
        metrics_dict['ece'].append(ece)
        metrics_dict['cece'].append(cece)
        
        #with h5py.File(resultspath, 'w') as hf:
        #    hf['mask'] = output

    df_metrics = pd.DataFrame(metrics_dict)
    df_metrics.loc['mean'] = df_metrics.mean().round(4)
    df_metrics.to_csv(metricspath)
