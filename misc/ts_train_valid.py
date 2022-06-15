import torch
import glob
from torch.utils.data import DataLoader
from calibrate.evaluation.metrics import ECELoss
from calibration_models_ts import *
from torch import nn, optim
import os
from tensorboardX import SummaryWriter
import time
import datetime
import os
import sys
import argparse
import random

from calibrate.net.unet import UNet
from common import *

sys.path.append(os.path.realpath(".."))

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', default=0, type=int, help='index of used GPU')
parser.add_argument('--model-name', default='LTS', type=str, help='model name: IBTS, LTS, TS')
parser.add_argument('--seg-model', default='svls', type=str, help='model name: ')
parser.add_argument('--dataset-type', default='promise_mc', type=str, help='dataset type: acdc, promise_mc, mrbrains, brats, flare')
parser.add_argument('--epochs', default=100, type=int, help='max epochs')
parser.add_argument('--batch-size', default=4, type=int, help='batch size')
parser.add_argument('--lr', default=1e-4, type=float, help='inital learning rate')
parser.add_argument('--seed', default=None, type=int, help='seed for initializing training.')
parser.add_argument('--save-per-epoch', default=1, type=int, help='number of epochs to save model.')


if __name__ == "__main__":

    args = parser.parse_args()
    
    model_name = str(args.model_name)
    seg_model_name = str(args.seg_model)
    dataset_type = str(args.dataset_type)
    max_epochs = args.epochs
    batch_size = args.batch_size
    lr = args.lr
    gpu = args.gpu
    
    segmodels = ['ce','ce_dice','focal','penalty','ls','svls','margin']
    params = get_params(dataset_type)
    
    model_path = params[segmodels.index(seg_model_name) + 3]
    print (model_path)
    data_root = params.data_root

    train_loader, val_loader =  params.train_loader, params.val_loader
    model = UNet(input_channels=params.in_channels, num_classes=params.nclasses)
    checkpoint = torch.load(model_path)["state_dict"]
    checkpoint = dict((key[7:] if "module" in key else key, value)for (key, value) in checkpoint.items())
    model.load_state_dict(checkpoint)
    for pms in model.parameters():
        pms.requires_grad = False 
        
    model = model.to(gpu)
    
    nll_criterion = nn.CrossEntropyLoss()
    # ece_criterion = ECELoss()
    
    
    if model_name == 'IBTS':
        experiment_name = model_name + '_CamVid' + '_epoch_' + str(max_epochs) + '_batchsize_' + str(batch_size) + '_lr_' + str(lr)
        calibration_model = IBTS_CamVid_With_Image()
    elif model_name == 'LTS':
        experiment_name = model_name + seg_model_name + '_CamVid' + '_epoch_' + str(max_epochs) + '_batchsize_' + str(batch_size) + '_lr_' + str(lr)
        calibration_model = LTS_CamVid_With_Image(input_channels=params.in_channels, num_classes = params.nclasses)
    elif model_name == 'TS':
        experiment_name = model_name + '_CamVid' + '_epoch_' + str(max_epochs) + '_batchsize_' + str(batch_size) + '_lr_' + str(lr)
        calibration_model = Temperature_Scaling()
    else:
        raise ValueError('Wrong Model Name!')


    calibration_model.weights_init()
    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        calibration_model.cuda(args.gpu)
    else:
        calibration_model.cuda()

    optimizer = optim.Adam(calibration_model.parameters(), lr=lr)

    print("Computing Loss")
    val_loss = 0
    # ece_loss = 0
    
    for val_image, val_labels in val_loader:
        
        val_logits = model(val_image.to(gpu))
        val_labels = val_labels.to(gpu)
        val_loss += nll_criterion(val_logits, val_labels).item()
        # ece_loss += ece_criterion(val_logits, val_labels).item()
        
    mean_val_loss = val_loss/len(val_loader)
    # mean_ece_loss = ece_loss/len(val_loader)

    print('Before calibration - NLL: %.5f' % (mean_val_loss))

    calibration_model.train()
    now = datetime.datetime.now()
    now_date = "{:02d}{:02d}{:02d}".format(now.month, now.day, now.year)
    now_time = "{:02d}{:02d}{:02d}".format(now.hour, now.minute, now.second)
    writer = SummaryWriter(os.path.join('./logs_CamVid', now_date, experiment_name + '_' + now_time))
    writer.add_scalar('validation/pre-post-nll',mean_val_loss,global_step=0)

    for epoch in range(max_epochs):
        for i, (train_image, train_labels) in enumerate(train_loader):
            global_step = epoch * len(train_loader) + (i + 1) * batch_size
            train_image, train_labels = train_image.cuda(gpu), train_labels.long().cuda(gpu)
            train_logits = model(train_image)
            
            optimizer.zero_grad()
            
            logits_calibrate = calibration_model(train_logits, train_image, args)
            loss = nll_criterion(logits_calibrate, train_labels)
            loss.backward()
            optimizer.step()
            print("{} epoch, {} iter, training loss: {:.5f}".format(epoch, i + 1, loss.item()))
            writer.add_scalar('loss/training', loss.item(), global_step=global_step)

            ## save the current best model and checkpoint
    # if i%10 == 9 and epoch % args.save_per_epoch == (args.save_per_epoch - 1):
        with torch.set_grad_enabled(False):
            tmp_loss = 0
            for val_image, val_labels, in val_loader:
                val_image, val_labels = val_image.cuda(gpu), val_labels.long().cuda(gpu)
                val_logits = model(val_image)
                logits_cali = calibration_model(val_logits, val_image, args)
                tmp_loss += nll_criterion(logits_cali, val_labels).item()
                
            mean_tmp_loss = tmp_loss/len(val_loader)
            print("{} epoch, {} iter, training loss: {:.5f}, val loss: {:.5f}".format(epoch, i+1, loss.item(), mean_tmp_loss))
            writer.add_scalar('loss/validation', mean_tmp_loss, global_step=global_step)
            # writer.add_image('loss/temperature', temperature, global_step=global_step)

            if mean_tmp_loss < mean_val_loss:
                mean_val_loss = mean_tmp_loss
                print('%d epoch, current lowest - NLL: %.5f' % (epoch, mean_val_loss))
                writer.add_scalar('validation/lowest_loss', mean_val_loss, global_step=global_step)
                torch.save(calibration_model.state_dict(), './calibration_Tiramisu/' + experiment_name + '_params.pth.tar')
                best_state = {'epoch': epoch,
                                'state_dict': calibration_model.state_dict(),
                                'optimizer': optimizer.state_dict(),
                                'best_score': mean_val_loss,
                                'global_step': global_step
                                }
                torch.save(best_state, './calibration_Tiramisu/' + experiment_name + '_model_best.pth.tar')

            current_state = {'epoch': epoch,
                                'state_dict': calibration_model.state_dict(),
                                'optimizer': optimizer.state_dict(),
                                'best_score': mean_tmp_loss,
                                'global_step': global_step
                            }
            torch.save(current_state, './calibration_Tiramisu/' + experiment_name + '_checkpoint.pth.tar')

    writer.add_scalar('validation/pre-post-nll',mean_val_loss,global_step=1)
