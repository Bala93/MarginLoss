from skimage.transform import resize
import numpy as np
from collections import namedtuple
from calibrate.data import *


def get_params(dataset_type, batch_size = 16, train_shuffle=True, valid_shuffle=False):
    
    temp_train_loader, temp_valid_loader, temp_test_loader = None, None, None
    test_loader = None
    
    if dataset_type == "acdc":

        in_channels = 1
        nclasses = 4
        data_root = '/home/ar88770/MarginLoss/acdc/'
                
        model_path_ce = '/home/ar88770/MarginLoss/outputs/cardiac/unet-ce-adam/20220618-10:33:34-326459/best.pth'
        model_path_ce_dice = '/home/ar88770/MarginLoss/outputs/cardiac/unet-ce_dice-adam/20220618-11:24:28-589840/best.pth'
        model_path_focal = '/home/ar88770/MarginLoss/outputs/cardiac/unet-focal-adam/20220619-19:26:40-615728/best.pth' ## 2
        model_path_penalty = '/home/ar88770/MarginLoss/outputs/cardiac/unet-penalty_ent-adam/20220619-14:55:16-193037/best.pth' ## 0.2
        model_path_ls = '/home/ar88770/MarginLoss/outputs/cardiac/unet-ls-adam/20220619-13:40:35-026779/best.pth' ## 0.2
        model_path_svls = '/home/ar88770/MarginLoss/outputs/cardiac/unet-svls-adam/20220618-15:03:57-522624/best.pth' ## 
        model_path_margin = '/home/ar88770/MarginLoss/outputs/cardiac/unet-logit_margin-adam/20220618-20:27:35-089776/best.pth' ## 
        model_path_adaptive_margin = '/home/ar88770/MarginLoss/outputs/cardiac/unet-adaptive_margin_svls-adam/20230214-10:11:12-549594/best.pth'
        
        train_loader, val_loader, _ = cardiac.get_train_val_loader(data_root, batch_size = batch_size)
        test_loader = cardiac.get_test_loader(data_root, batch_size = batch_size)

    if dataset_type == "flare":

        in_channels = 1
        nclasses = 5
        data_root = '/home/ar88770/MarginLoss/flare/'
        
        model_path_ce = '/home/ar88770/MarginLoss/outputs/abdomen/unet-ce-adam/20220619-21:08:32-970895/best.pth'
        model_path_ce_dice = '/home/ar88770/MarginLoss/outputs/abdomen/unet-ce_dice-adam/20220620-04:05:56-057493/best.pth'
        model_path_focal = '/home/ar88770/MarginLoss/outputs/abdomen/unet-focal-adam/20220622-10:26:48-012082/best.pth'
        model_path_penalty = '/home/ar88770/MarginLoss/outputs/abdomen/unet-penalty_ent-adam/20220620-13:38:47-558393/best.pth'
        model_path_ls = '/home/ar88770/MarginLoss/outputs/abdomen/unet-ls-adam/20220620-21:49:13-261176/best.pth'
        model_path_svls = '/home/ar88770/MarginLoss/outputs/abdomen/unet-svls-adam/20220621-03:56:48-576349/best.pth'
        model_path_margin = '/home/ar88770/MarginLoss/outputs/abdomen/unet-logit_margin-adam/20220624-05:40:30-578821/best.pth'
        model_path_adaptive_margin = '/home/ar88770/MarginLoss/outputs/abdomen/unet-adaptive_margin_svls-adam/20230213-20:29:23-871237/best.pth'
        
        train_loader, val_loader, _ = abdomen.get_train_val_loader(data_root, batch_size = batch_size)
        test_loader = abdomen.get_test_loader(data_root, batch_size = batch_size)
        
        
    if dataset_type == 'brats19':

        in_channels = 4
        nclasses = 4
        data_root = '/home/ar88770/MarginLoss/brats19/'
        model_path_ce = '/home/ar88770/MarginLoss/outputs/brain19/unet-ce-adam/20221228-15:17:39-946670/best.pth'
        model_path_ce_dice = '/home/ar88770/MarginLoss/outputs/brain19/unet-ce_dice-adam/20221228-22:37:12-150988/best.pth'
        model_path_focal = '/home/ar88770/MarginLoss/outputs/brain19/unet-focal-adam/20221229-08:31:19-541909/best.pth'
        model_path_penalty = '/home/ar88770/MarginLoss/outputs/brain19/unet-penalty_ent-adam/20221231-20:23:28-289240/best.pth'
        model_path_ls = '/home/ar88770/MarginLoss/outputs/brain19/unet-ls-adam/20221229-22:43:51-938934/best.pth'
        model_path_svls = '/home/ar88770/MarginLoss/outputs/brain19/unet-svls-adam/20221229-22:27:33-048617/best.pth'
        model_path_margin = '/home/ar88770/MarginLoss/outputs/brain19/unet-logit_margin-adam/20221231-20:24:31-370235/best.pth'
        model_path_adaptive_margin = '/home/ar88770/MarginLoss/outputs/brain19/unet-adaptive_margin_svls-adam/20230213-22:50:42-413710/best.pth'

        train_loader, val_loader, _ = brain19.get_train_val_loader(data_root, batch_size = batch_size, train_shuffle=False, valid_shuffle=False)
        test_loader = brain19.get_test_loader(data_root, batch_size = 128)
        temp_train_loader, temp_valid_loader, temp_test_loader = brain19.get_post_temp_scaling_loader(data_root, batch_size = batch_size)
            
    if dataset_type == 'hcmps':

        in_channels = 1
        nclasses = 3
        data_root = '/home/ar88770/MarginLoss/hcmps/'
        model_path_ce = '/home/ar88770/MarginLoss/outputs/hcmps/unet-ce-adam/20220620-16:30:12-885599/best.pth'
        model_path_ce_dice = '/home/ar88770/MarginLoss/outputs/hcmps/unet-ce_dice-adam/20220620-17:11:50-852909/best.pth'
        model_path_focal = '/home/ar88770/MarginLoss/outputs/hcmps/unet-focal-adam/20220621-17:40:33-503567/best.pth'
        model_path_penalty = '/home/ar88770/MarginLoss/outputs/hcmps/unet-penalty_ent-adam/20220620-18:36:36-312382/best.pth'
        model_path_ls = '/home/ar88770/MarginLoss/outputs/hcmps/unet-ls-adam/20220620-19:18:59-720043/best.pth'
        model_path_svls = '/home/ar88770/MarginLoss/outputs/hcmps/unet-svls-adam/20220620-20:01:16-191480/best.pth'
        model_path_margin = '/home/ar88770/MarginLoss/outputs/hcmps/unet-logit_margin-adam/20220620-21:08:29-180117/best.pth'

        train_loader, val_loader, _ = hcmps.get_train_val_loader(data_root, batch_size = batch_size, train_shuffle=True, valid_shuffle=True)
        test_loader = hcmps.get_test_loader(data_root, batch_size = 32)
        
        temp_train_loader, temp_valid_loader, temp_test_loader = hcmps.get_post_temp_scaling_loader(data_root, batch_size = batch_size)
        
    if dataset_type == 'busi':
        
        in_channels = 1
        nclasses = 2
        data_root = '/home/ar88770/MarginLoss/busi/'
        
        model_path_ce = '/home/ar88770/MarginLoss/outputs/busi/unet-ce-adam/20221230-22:02:51-494146/best.pth'
        model_path_ce_dice = '/home/ar88770/MarginLoss/outputs/busi/unet-ce_dice-adam/20221230-22:32:42-707506/best.pth'
        model_path_focal = '/home/ar88770/MarginLoss/outputs/busi/unet-focal-adam/20221230-23:05:39-227975/best.pth'
        model_path_penalty = '/home/ar88770/MarginLoss/outputs/busi/unet-penalty_ent-adam/20221231-11:35:16-467991/best.pth'
        model_path_ls = '/home/ar88770/MarginLoss/outputs/busi/unet-ls-adam/20221231-12:08:32-005817/best.pth'
        model_path_svls = '/home/ar88770/MarginLoss/outputs/busi/unet-svls-adam/20221231-00:43:02-508661/best.pth' 
        model_path_margin = '/home/ar88770/MarginLoss/outputs/busi/unet-logit_margin-adam/20230109-11:34:40-630663/best.pth'
        
        train_loader, val_loader, _ = busi.get_train_val_loader(data_root, batch_size = batch_size)
        test_loader = busi.get_test_loader(data_root, batch_size = 16)
        
        temp_train_loader, temp_valid_loader, temp_test_loader = busi.get_post_temp_scaling_loader(data_root, batch_size = batch_size)
        
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
        
        train_loader, val_loader, _ = brainatlas.get_train_val_loader(data_root, batch_size = batch_size)
        
    if dataset_type == "promise_mc":
        
        in_channels = 1
        nclasses = 3
        
        data_root = '/home/ar88770/MarginLoss/promise_mc/'
        model_path_ce = '/home/ar88770/MarginLoss/outputs/prostate_mc/unet-ce-adam/20220618-19:45:47-605216/best.pth'
        model_path_ce_dice = '/home/ar88770/MarginLoss/outputs/prostate_mc/unet-ce_dice-adam/20220618-19:59:27-919203/best.pth'
        model_path_focal = '/home/ar88770/MarginLoss/outputs/prostate_mc/unet-focal-adam/20220619-19:27:36-214693/best.pth'
        model_path_penalty = '/home/ar88770/MarginLoss/outputs/prostate_mc/unet-penalty_ent-adam/20220618-20:26:38-657056/best.pth'
        model_path_ls = '/home/ar88770/MarginLoss/outputs/prostate_mc/unet-ls-adam/20220619-18:30:02-915430/best.pth'
        model_path_svls = '/home/ar88770/MarginLoss/outputs/prostate_mc/unet-svls-adam/20220618-20:55:27-830091/best.pth'
        model_path_margin = '/home/ar88770/MarginLoss/outputs/prostate_mc/unet-logit_margin-adam/20220618-21:09:31-193817/best.pth'
        
        train_loader, val_loader, _ = prostate_mc.get_train_val_loader(data_root, batch_size = batch_size)
        
    
    params = {'in_channels':in_channels, 'nclasses': nclasses, 'data_root': data_root, 
              'model_path_ce': model_path_ce, 'model_path_ce_dice': model_path_ce_dice,
              'model_path_focal': model_path_focal, 'model_path_penalty': model_path_penalty,
              'model_path_ls': model_path_ls, 'model_path_svls': model_path_svls,
              'model_path_margin': model_path_margin,
              'model_path_adpt': model_path_adaptive_margin,
              'train_loader': train_loader,
              'val_loader': val_loader,
              'test_loader': test_loader,
              't_train_loader':temp_train_loader, 
              't_valid_loader':temp_valid_loader, 
              't_test_loader':temp_test_loader}
    
    
    return namedtuple("ObjectName", params.keys())(*params.values())


def pre_process_data(img, mask, dataset_type, sno=None):
    

    if dataset_type == "flare":
        
        if sno is  None:
            img = np.expand_dims(img,1)
            mask = mask
        else:
            img = img[sno]
            mask = mask[sno]
        
    if dataset_type == "brats19":
        
        if sno is  None:
            img = img.transpose(1,0,2,3)
            mask = mask.transpose(1,0,2,3)[:,0]
        else:
            img = img[:,sno]
            mask = mask[sno]
        
    if dataset_type == "acdc":
        if sno is None:
            img = img[16:208,16:208,:]
            mask = mask[16:208,16:208,:]
            img = np.transpose(img, axes=[2,0,1])
            img = np.expand_dims(img, 1)
            mask = np.transpose(mask, axes=[2,0,1])
            
        else:            
            img = img[16:208,16:208,sno]
            mask = mask[16:208,16:208,sno]
            img = np.expand_dims(img, axis=0)          
    
    if dataset_type == "refuge":
        
        mask[mask==0] = 2
        mask[mask==128] = 1
        mask[mask==255] = 0

        img = resize(img,[256,256],order=3,preserve_range=True)
        mask = resize(mask, [256,256],order=0,preserve_range=True)               
        img = np.transpose(img, axes=[2,0,1])

    if dataset_type == "mrbrains":

        img = np.pad(img,pad_width=((0,0),(8,8),(8,8),(0,0)),mode='constant') # image = image[:,24:216,24:216]
        mask = np.pad(mask,pad_width=((8,8),(8,8),(0,0)),mode='constant') # mask = mask[24:216,24:216]
        mask[mask==4] = 0 
        
        if not sno is None:        
            img = img[:,:,:,sno] 
            mask = mask[:,:,sno]
        
    if dataset_type == "promise_mc":

        img = img[:,80:240,80:240]
        mask = mask[:,80:240,80:240]
        img = img[sno,:,:] 
        mask = mask[sno,:,:]  
        
        img = np.expand_dims(img, axis=0)          
            
    return img, mask
