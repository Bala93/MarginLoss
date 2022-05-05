import glob
from calibrate.net.unet import UNet
from calibrate.net.vit_seg_modeling import transfomer_model
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
from segmentation_models_pytorch import UnetPlusPlus


data_root = '/home/ar88770/MarginLoss/flare/valid'
# unet 
# model_path_ce = '/home/ar88770/MarginLoss/outputs/abdomen/unet-ce-adam/20220501-23:43:11-336368/best.pth'
# model_path_margin = '/home/ar88770/MarginLoss/outputs/abdomen/unet-logit_margin-adam/20220502-07:01:38-850966/best.pth'
## unetpp
# model_path_ce = '/home/ar88770/MarginLoss/outputs/abdomen/unetpp-ce-adam/20220503-08:58:27-961344/best.pth'
# model_path_margin = '/home/ar88770/MarginLoss/outputs/abdomen/unetpp-logit_margin-adam/20220503-12:56:36-184488/best.pth'
## transunet
model_path_ce = '/home/ar88770/MarginLoss/outputs/abdomen/transunet-ce-adam/20220503-17:50:32-275714/best.pth'
model_path_margin = '/home/ar88770/MarginLoss/outputs/abdomen/transunet-logit_margin-adam/20220504-08:48:50-840940/best.pth'



in_channels = 1
nclasses = 5
method_names =['ce','margin']

files = glob.glob('{}/*.h5'.format(data_root))
models_path = [model_path_ce, model_path_margin]
# model = UNet(input_channels=in_channels, num_classes=nclasses)
# model = UnetPlusPlus(in_channels=in_channels, classes=nclasses)
model = transfomer_model(model_name='R50-ViT-B_16', img_size=192, vit_patches_size=16, n_classes=nclasses)

for _, model_path in zip(method_names,models_path):

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
            imgT = torch.from_numpy(img)
            imgT = imgT.unsqueeze(0).unsqueeze(0)
            imgT = imgT.cuda('cuda:4')
            
            mask = volmask[sno, :,:]            

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