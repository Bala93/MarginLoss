import glob
from calibrate.net.unet import UNet
import numpy as np
import torch
from tqdm import tqdm
import h5py
from torch.nn import functional as F
import os
import pandas as pd
from utils import *
from common import *
from scipy.ndimage import morphology
import argparse

def get_metric_name(ece_choice):
    

    if ece_choice == 'fg': 
        metric_fname = 'metrics3d_fg.csv'
    elif ece_choice == 'fgbnd': 
        metric_fname = 'metrics3d_fg_dil.csv'
    elif ece_choice == 'bnd': 
        metric_fname = 'metrics3d_bnd.csv'
    else:
        raise ValueError("Provide valid choice")
        
    return metric_fname 

def get_target_for_calib(target, ece_choice):
    
    if ece_choice == 'fg': 
        utarget = target 
    elif ece_choice == 'fgbnd': 
        utarget = morphology.binary_dilation(target, iterations=2).astype(int)
    elif ece_choice == 'bnd': 
        utarget = morphology.binary_dilation(target, iterations=2).astype(int)
        utarget = np.logical_xor(target, utarget).astype(int)
    else:
        raise ValueError("Provide valid choice")

    return utarget
    
    

if __name__ == "__main__":
    

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--dataset_type',type=str,required=True)
    parser.add_argument('--model_path',type=str,required=True)
    parser.add_argument('--ece_choice',type=str,default='fg')
    
    args = parser.parse_args()
    
    dataset_type = args.dataset_type
    ece_choice = args.ece_choice
    model_path = args.model_path
    
    device = 'cuda:4'
    to_save_pred = False
    mode = 'test'
    is_shape_metrics = True
    
    params = get_params(dataset_type)

    model = UNet(input_channels=params.in_channels, num_classes=params.nclasses)
    metric_fname = get_metric_name(ece_choice)
    
    files = glob.glob('{}/{}/*.h5'.format(params.data_root,mode))

    checkpoint = torch.load(model_path)["state_dict"]
    checkpoint = dict((key[7:] if "module" in key else key, value)for (key, value) in checkpoint.items())
    model.load_state_dict(checkpoint)
    model = model.cuda(device)
    
    metrics_dict = {"fname":[],"dsc":[],"hd":[], "ece":[],"cece":[], "nll":[]}
    savedir = os.path.dirname(model_path)
    metricspath = os.path.join(savedir, metric_fname)       
    
    for fpath in tqdm(files):
        
        fname = os.path.basename(fpath)
    
        with h5py.File(fpath, 'r') as data:

            img = data["img"][:]
            target = data["mask"][:]
            
        # print (img.shape, target.shape)

        img, target = pre_process_data(img, target, dataset_type)
        targetc = get_target_for_calib(target, ece_choice)
        
        imgT = torch.from_numpy(img).float().cuda(device)
        targetT = torch.from_numpy(targetc).cuda(device)

        predT = model(imgT)
                    
        outputconf = F.softmax(predT,dim=1).detach().cpu().numpy()
        output = np.argmax(outputconf,axis=1)
        
        if is_shape_metrics:
            dsc, hd = shape_metrics(output, target, params.nclasses, fpath)
        else:
            dsc, hd = 1, 0 

        n, c, x, y = predT.shape
        logits = torch.einsum("ncxy->nxyc", predT)
        logits = logits.reshape(n * x * y, -1)
        labels = targetT.reshape(n * x * y)
        output = output.reshape(n * x * y)
        
        index = torch.nonzero(labels != 0).squeeze()
        
        logits = logits[index, :]
        labels = labels[index]

        ece, cece, aece, nll = calib_metrics(logits, labels)
    
        metrics_dict['fname'].append(fname)
        metrics_dict['dsc'].append(dsc)
        metrics_dict['hd'].append(hd)
        metrics_dict['ece'].append(ece)
        metrics_dict['cece'].append(cece)
        metrics_dict['nll'].append(nll)

        if to_save_pred:
            resultsdir = os.path.join(savedir, 'results') 
            if os.path.exists(resultsdir):
                os.mkdir(resultsdir)
            resultspath = os.path.join(resultsdir, fname)
            with h5py.File(resultspath, 'w') as hf:
                hf['mask'] = output
                
        # break

    df_metrics = pd.DataFrame(metrics_dict)
    df_metrics.loc['mean'] = df_metrics.mean().round(4)
    df_metrics.to_csv(metricspath)
    
    # break
