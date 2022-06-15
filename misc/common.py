from skimage.transform import resize
import numpy as np
from collections import namedtuple
from calibrate.data import *


def get_params(dataset_type):
    
    if dataset_type == "refuge":

        in_channels = 3
        nclasses = 3
        data_root = '/home/ar88770/MarginLoss/refuge/'
        model_path_ce = '/home/ar88770/MarginLoss/outputs/fundus/unet-ce-adam/20220401-16:19:09-606441/best.pth'
        model_path_ce_dice = '/home/ar88770/MarginLoss/outputs/fundus/unet-ce_dice-adam/20220427-11:21:48-510735/best.pth'
        model_path_focal = '/home/ar88770/MarginLoss/outputs/fundus/unet-focal-adam/20220406-16:17:33-121435/best.pth'
        model_path_penalty = '/home/ar88770/MarginLoss/outputs/fundus/unet-penalty_ent-adam/20220406-17:10:29-709839/best.pth'
        model_path_ls = '/home/ar88770/MarginLoss/outputs/fundus/unet-ls-adam/20220401-22:06:25-082822/best.pth'
        model_path_svls = '/home/ar88770/MarginLoss/outputs/fundus/unet-svls-adam/20220417-14:08:54-166306/best.pth' 
        model_path_margin = '/home/ar88770/MarginLoss/outputs/fundus/unet-logit_margin-adam/20220401-23:15:07-350192/best.pth'
        model_path_adaptive_margin = '/home/ar88770/MarginLoss/outputs/fundus/unet-logit_margin_adaptive-adam/20220522-09:16:43-247458/best.pth'

    if dataset_type == "flare":

        in_channels = 1
        nclasses = 5
        data_root = '/home/ar88770/MarginLoss/flare/'
        model_path_ce = '/home/ar88770/MarginLoss/outputs/abdomen/unet-ce-adam/20220501-23:43:11-336368/best.pth'
        model_path_ce_dice = '/home/ar88770/MarginLoss/outputs/abdomen/unet-ce_dice-adam/20220502-18:52:50-928615/best.pth'
        model_path_focal = '/home/ar88770/MarginLoss/outputs/abdomen/unet-focal-adam/20220502-09:49:29-657707/best.pth'
        model_path_penalty = '/home/ar88770/MarginLoss/outputs/abdomen/unet-penalty_ent-adam/20220502-12:21:41-348902/best.pth'
        model_path_ls = '/home/ar88770/MarginLoss/outputs/abdomen/unet-ls-adam/20220502-04:02:05-859276/best.pth'
        model_path_svls = '/home/ar88770/MarginLoss/outputs/abdomen/unet-svls-adam/20220502-15:09:37-710891/best.pth'
        model_path_margin = '/home/ar88770/MarginLoss/outputs/abdomen/unet-logit_margin-adam/20220502-07:01:38-850966/best.pth'
        model_path_adaptive_margin = '/home/ar88770/MarginLoss/outputs/abdomen/unet-logit_margin_adaptive-adam/20220522-09:16:38-137376/best.pth'
        train_loader, val_loader, _ = abdomen.get_train_val_loader(data_root)
        
    if dataset_type == "brats":
        
        in_channels = 4
        nclasses = 4
        data_root = '/home/ar88770/MarginLoss/brats/'
        model_path_ce = '/home/ar88770/MarginLoss/outputs/brain/unet-ce-adam/20220327-22:46:38-903888/best.pth'
        model_path_ce_dice = '/home/ar88770/MarginLoss/outputs/brain/unet-ce_dice-adam/20220427-15:57:36-359885/best.pth'
        model_path_focal = '/home/ar88770/MarginLoss/outputs/brain/unet-focal-adam/20220405-15:44:14-601824/best.pth'
        model_path_penalty = '/home/ar88770/MarginLoss/outputs/brain/unet-penalty_ent-adam/20220405-23:10:20-562585/best.pth'
        model_path_ls = '/home/ar88770/MarginLoss/outputs/brain/unet-ls-adam/20220328-02:43:30-442461/best.pth'
        model_path_svls = '/home/ar88770/MarginLoss/outputs/brain/unet-svls-adam/20220417-14:12:05-491765/best.pth'
        model_path_margin = '/home/ar88770/MarginLoss/outputs/brain/unet-logit_margin-adam/20220328-06:40:26-544931/best.pth'
        model_path_adaptive_margin = '/home/ar88770/MarginLoss/outputs/brain/unet-logit_margin_adaptive-adam/20220522-09:16:35-042308/best.pth'
        train_loader, val_loader, _ = brain.get_train_val_loader(data_root)
        
    if dataset_type == "acdc":

        in_channels = 1
        nclasses = 4
        data_root = '/home/ar88770/MarginLoss/acdc/'
        model_path_ce = '/home/ar88770/MarginLoss/outputs/cardiac/unet-ce-adam/20220323-20:17:06-951406/best.pth'
        model_path_ce_dice = '/home/ar88770/MarginLoss/outputs/cardiac/unet-ce_dice-adam/20220427-13:35:35-177337/best.pth'
        model_path_focal = '/home/ar88770/MarginLoss/outputs/cardiac/unet-focal-adam/20220323-21:00:04-839009/best.pth'
        model_path_penalty = '/home/ar88770/MarginLoss/outputs/cardiac/unet-penalty_ent-adam/20220323-22:46:02-093977/best.pth'
        model_path_ls = '/home/ar88770/MarginLoss/outputs/cardiac/unet-ls-adam/20220324-17:38:33-210962/best.pth'
        model_path_svls = '/home/ar88770/MarginLoss/outputs/cardiac/unet-svls-adam/20220417-12:03:39-233516/best.pth'
        model_path_margin = '/home/ar88770/MarginLoss/outputs/cardiac/unet-logit_margin-adam/20220324-14:58:28-642625/best.pth'
        model_path_adaptive_margin = '/home/ar88770/MarginLoss/outputs/cardiac/unet-logit_margin_adaptive-adam/20220522-09:16:31-063742/best.pth'
        train_loader, val_loader, _ = cardiac.get_train_val_loader(data_root,batch_size=4)
        
    if dataset_type == "mrbrains":

        in_channels = 3
        nclasses = 4     
        data_root = '/home/ar88770/MarginLoss/mrbrains/'
        model_path_ce = '/home/ar88770/MarginLoss/outputs/brainatlas/unet-ce-adam/20220404-16:20:24-829672/best.pth'
        model_path_ce_dice = '/home/ar88770/MarginLoss/outputs/brainatlas/unet-ce_dice-adam/20220427-17:25:09-405389/best.pth'
        model_path_focal = '/home/ar88770/MarginLoss/outputs/brainatlas/unet-focal-adam/20220406-10:59:04-103885/best.pth'
        model_path_penalty = '/home/ar88770/MarginLoss/outputs/brainatlas/unet-penalty_ent-adam/20220406-11:26:33-769991/best.pth'
        model_path_ls = '/home/ar88770/MarginLoss/outputs/brainatlas/unet-ls-adam/20220404-16:49:47-579491/best.pth'
        model_path_svls = '/home/ar88770/MarginLoss/outputs/brainatlas/unet-svls-adam/20220417-17:23:24-173305/best.pth'
        model_path_margin = '/home/ar88770/MarginLoss/outputs/brainatlas/unet-logit_margin-adam/20220404-22:15:12-835794/best.pth'
        model_path_adaptive_margin = '/home/ar88770/MarginLoss/outputs/brainatlas/unet-logit_margin_adaptive-adam/20220522-10:06:32-367361/best.pth'
        train_loader, val_loader, _ = brainatlas.get_train_val_loader(data_root)
        
    if dataset_type == "promise_mc":
        
        in_channels = 1
        nclasses = 3
        data_root = '/home/ar88770/MarginLoss/promise_mc/'
        model_path_ce = '/home/ar88770/MarginLoss/outputs/prostate_mc/unet-ce-adam/20220409-22:47:25-847975/best.pth'
        model_path_ce_dice = '/home/ar88770/MarginLoss/outputs/prostate_mc/unet-ce_dice-adam/20220427-13:38:39-924288/best.pth'
        model_path_focal = '/home/ar88770/MarginLoss/outputs/prostate_mc/unet-focal-adam/20220406-17:06:46-112063/best.pth'
        model_path_penalty = '/home/ar88770/MarginLoss/outputs/prostate_mc/unet-penalty_ent-adam/20220411-23:10:20-338365/best.pth'
        model_path_ls = '/home/ar88770/MarginLoss/outputs/prostate_mc/unet-ls-adam/20220409-22:57:05-534513/best.pth'
        model_path_svls = '/home/ar88770/MarginLoss/outputs/prostate_mc/unet-svls-adam/20220417-14:10:04-419455/best.pth'
        model_path_margin = '/home/ar88770/MarginLoss/outputs/prostate_mc/unet-logit_margin-adam/20220409-23:07:04-480575/best.pth'
        model_path_adaptive_margin = '/home/ar88770/MarginLoss/outputs/prostate_mc/unet-logit_margin_adaptive-adam/20220522-09:16:40-812822/best.pth'
        train_loader, val_loader, _ = prostate_mc.get_train_val_loader(data_root)
        
    
    params = {'in_channels':in_channels, 'nclasses': nclasses, 'data_root': data_root, 
              'model_path_ce': model_path_ce, 'model_path_ce_dice': model_path_ce_dice,
              'model_path_focal': model_path_focal, 'model_path_penalty': model_path_penalty,
              'model_path_ls': model_path_ls, 'model_path_svls': model_path_svls,
              'model_path_margin': model_path_margin,
              'model_path_adpt': model_path_adaptive_margin,
              'train_loader': train_loader,
              'val_loader': val_loader
              }
    
    
    return namedtuple("ObjectName", params.keys())(*params.values())


def pre_process_data(img, mask, dataset_type, sno):
    
    if dataset_type == "refuge":
        
        mask[mask==0] = 2
        mask[mask==128] = 1
        mask[mask==255] = 0

        img = resize(img,[256,256],order=3,preserve_range=True)
        mask = resize(mask, [256,256],order=0,preserve_range=True)               
        img = np.transpose(img, axes=[2,0,1])

    if dataset_type == "flare":
        
        img = img[sno]
        mask = mask[sno]
        img = np.expand_dims(img,axis=0)
        
    if dataset_type == "brats":
        
        img = np.pad(img,pad_width=((0,0),(0,0),(8,8),(8,8)),mode='constant')
        mask = np.pad(mask,pad_width=((0,0),(8,8),(8,8)),mode='constant')
        img = img[:,sno]
        mask = mask[sno]
        
    if dataset_type == "acdc":

        img = np.pad(img,pad_width=((5,5),(5,5)),mode='constant')
        mask = np.pad(mask,pad_width=((5,5),(5,5)),mode='constant')
        img = np.expand_dims(img, axis=0)
        
    if dataset_type == "mrbrains":

        img = np.pad(img,pad_width=((0,0),(8,8),(8,8),(0,0)),mode='constant')
        mask = np.pad(mask,pad_width=((8,8),(8,8),(0,0)),mode='constant')
        mask[mask==4] = 0 
        
        img = img[:,:,:,sno] 
        mask = mask[:,:,sno]
        
    if dataset_type == "promise_mc":

        img = img[:,80:240,80:240]
        mask = mask[:,80:240,80:240]
        img = img[sno,:,:] 
        mask = mask[sno,:,:]  
        
        img = np.expand_dims(img, axis=0)          
            
    return img, mask
