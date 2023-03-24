import os.path as osp
import numpy as np
from typing import Dict
import time
import json
import logging
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
import wandb
from terminaltables.ascii_table import AsciiTable
from typing import Optional
from calibrate.net import ModelWithTemperature, LTS_CamVid_With_Image, Temperature_Scaling, LTS_LW
# from calibrate.losses import LabelSmoothConstrainedLoss
from calibrate.evaluation import (
    AverageMeter, LossMeter, SegmentEvaluator, SegmentCalibrateEvaluator, CalibSegmentEvaluator
)
from calibrate.utils import (
    load_train_checkpoint, load_checkpoint, save_checkpoint, round_dict
)
from calibrate.utils.file_io import mkdir
from calibrate.utils.torch_helper import entropy, to_numpy, get_lr
from .tester import Tester
from calibrate.utils.misc import bratspostprocess
from tqdm import tqdm
from torch import nn
from calibrate.evaluation.metrics import ECELoss
from calibrate.net.temperature_scaling import Temperature_Scaling, Local_Temperature_Scaling

logger = logging.getLogger(__name__)


class MedSegmentTester(Tester):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)

    def build_meter(self):
        self.batch_time_meter = AverageMeter()
        self.num_classes = self.cfg.model.num_classes
        self.evaluator = CalibSegmentEvaluator(
            self.test_loader.dataset.classes,
            ignore_index=255,
            ishd=True,
            dataset_type=self.cfg.data.name)
        
    @torch.no_grad()
    def eval_epoch(self, data_loader, phase="Val",post_temp=False,ts_type='ts'):
        self.reset_meter()
        self.model.eval()

        end = time.time()
        # fsave = open(osp.join(self.work_dir, "segment_{}_calibrate.txt".format(self.cfg.loss.name)), "w")
        for i, (inputs, labels, fpath) in enumerate(tqdm(data_loader)):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            inputs = inputs.squeeze()
            labels = labels.squeeze()
            
            # print (fpath)
            
            if len(inputs.shape) == 3:
                inputs = inputs.unsqueeze(1)
            
            if len(inputs.shape) == 2:
                inputs = inputs.unsqueeze(0).unsqueeze(0)
                labels = labels.unsqueeze(0)
            
            # forward
            outputs = self.model(inputs)
            if isinstance(outputs, Dict):
                outputs = outputs["out"]
                
            if post_temp:
                if ts_type == 'grid':
                    # outputs = outputs / self.temperature/
                    outputs = self.ts_model(outputs, inputs, self.device)
                elif ts_type == 'ts':
                    outputs = self.ts_model(outputs)
                elif ts_type == 'lts':
                    outputs = self.ts_model(outputs,inputs, self.device)
            # metric
            predicts = F.softmax(outputs, dim=1)
            pred_labels = torch.argmax(predicts, dim=1)
            
            ## to convert the brats to the metric setup
            # if self.cfg.data.name == 'brain':
            #     pred_labels, labels = bratspostprocess(pred_labels, labels)
            
            self.evaluator.update(
                to_numpy(pred_labels),
                to_numpy(labels),
                outputs,
                labels,
                fpath
            )
            
            # self.calibrate_evaluator.update(
            #     outputs, labels
            # )
            
            # self.logits_evaluator(
            #     np.expand_dims(to_numpy(outputs), axis=0),
            #     np.expand_dims(to_numpy(labels), axis=0)
            # )
            # measure elapsed time
            self.batch_time_meter.update(time.time() - end)
            # logging
            # if (i + 1) % self.cfg.log_period == 0:
            #     self.log_iter_info(i, max_iter, epoch, phase)
            end = time.time()
            # logger.info("\n" + "Processing {}".format(i))
        #     if self.cfg.test.save_logits:
        #         logits_save_dir = osp.join(self.work_dir, "segment_{}".format(self.cfg.loss.name))
        #         mkdir(logits_save_dir)
        #         for j, name in enumerate(sample_id):
        #             logger.info("save result for {}".format(name))
        #             save_path = osp.join(logits_save_dir, name + ".npz")
        #             np.savez(
        #                 save_path,
        #                 image=to_numpy(inputs[j]),
        #                 logit=to_numpy(outputs[j]),
        #                 predict=to_numpy(predicts[j]),
        #                 label=to_numpy(labels[j])
        #             )
        #             fsave.write("{}\t{:.5f}\n".format(name, ece))
        # fsave.close()
        self.log_eval_epoch_info(phase)

    def log_eval_epoch_info(self, phase="Val"):
        log_dict = {}
        #log_dict["samples"] = self.evaluator.num_samples()
        metric = self.evaluator.mean_score()
        log_dict.update(metric)
        # calibrate_metric, calibrate_table_data = self.calibrate_evaluator.mean_score(isprint=False)
        # log_dict.update(calibrate_metric)
        logger.info("{} Epoch\t{}".format(
            phase, json.dumps(round_dict(log_dict))
        ))
        class_table_data = self.evaluator.class_score(isprint=True, return_dataframe=True)
        calibrate_table_data = self.evaluator.calib_score(isprint=True)
        
        class_hd_list = class_table_data['hd'].to_list()[:-1]
        class_dice_list = class_table_data['dsc'].to_list()[:-1]
        class_name_list = class_table_data['Class'].to_list()[:-1]
        
        print (self.cfg.test.checkpoint)
        self.evaluator.save_csv(osp.dirname(str(self.cfg.test.checkpoint)))
        
        for ii in range(len(class_name_list)):
            key = 'dsc-{}'.format(class_name_list[ii])
            val = class_dice_list[ii]
            log_dict.update({key:val})    
            
            key = 'hd-{}'.format(class_name_list[ii])
            val = class_hd_list[ii]
            log_dict.update({key:val})
        
        if self.cfg.wandb.enable:
            wandb_log_dict = {}
            wandb_log_dict.update(dict(
                ("{}/{}".format(phase, key), value) for (key, value) in log_dict.items()
            ))
            wandb_log_dict["{}/segment_score_table".format(phase)] = (
                wandb.Table(
                    dataframe=class_table_data
                )
            )
            if phase.lower() == "test":
                wandb_log_dict["{}/calibrate_score_table".format(phase)] = (
                    wandb.Table(
                        dataframe=calibrate_table_data
                    )
                )
            wandb.log(wandb_log_dict)
            

    @torch.no_grad()
    def find_best_temperature(self, data_loader):
        
        self.model.eval()
        
        T = 0.1
        
        nll_criterion = nn.CrossEntropyLoss() 
        #ECELoss()
        nll_best = 1e10
        
        # niter = len(data_loader) 
        
        for j in tqdm(range(100)):
            
            nll_post = 0
            cnt = 0
            
            for i, (inputs, labels) in enumerate(data_loader):
                inputs, targets = inputs.to(self.device), labels.to(self.device)                                 
                
                logits = self.model(inputs)
                
                if isinstance(logits, Dict):
                    logits = logits["out"]
                    
                logits = logits / T
                                
                nll_post += nll_criterion(logits, targets).detach().cpu().numpy()        
                cnt  += 1
                
                # if i == niter:
                #     break
                    
            nll_post_mean = nll_post / cnt
            
            if nll_post_mean < nll_best:
                nll_best = nll_post_mean
                postT = T
            
            T += 0.1
            
        return postT

    def find_best_temperature_optimize(self, data_loader):
        
        self.model.eval()
        
        nll_criterion = nn.CrossEntropyLoss() 
        
        nll_pre = 0
        
        ## before temperature scaling
        for i, (inputs, labels) in enumerate(data_loader):
            inputs, targets = inputs.to(self.device), labels.to(self.device)                                 
            logits = self.model(inputs)
            
            if isinstance(logits, Dict):
                logits = logits["out"]
                            
            nll_pre += nll_criterion(logits, targets).detach().cpu().numpy()        

        nll_best = nll_pre / len(data_loader)
        
        # temp_model = Temperature_Scaling().to(self.device)
        temp_model = Local_Temperature_Scaling(input_channels=1, num_classes=4).to(self.device)
        ## optimizing for temperature
        optimizer =  torch.optim.Adam(temp_model.parameters(), lr=1e-4)
        
        for j in tqdm(range(50)):
            
            # update
            for i, (inputs, labels) in enumerate(data_loader):
                inputs, targets = inputs.to(self.device), labels.to(self.device)                                 
                
                logits = temp_model(self.model(inputs), inputs,self.device)
                
                if isinstance(logits, Dict):
                    logits = logits["out"]
                    
                # output = logits / temperature
                loss = nll_criterion(logits, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
             
            # after temperature scaling
            nll_post = 0
            for i, (inputs, labels) in enumerate(data_loader):
                inputs, targets = inputs.to(self.device), labels.to(self.device)                                 
                logits = temp_model(self.model(inputs), inputs, self.device)
                
                if isinstance(logits, Dict):
                    logits = logits["out"]
                                
                nll_post += nll_criterion(logits, targets).detach().cpu().numpy()        
   
            nll_post_mean = nll_post / len(data_loader)
            
            if nll_post_mean < nll_best:
                nll_best = nll_post_mean
            
            # postT = temp_model.get_temperature()
                        
        return temp_model


    def test(self):
        logger.info(
            "Everything is perfect so far. Let's start testing. Good luck!"
        )
        # temperature = 1
        if self.cfg.test.post_temperature:
            if self.cfg.test.ts_type == 'ts':
                                
                self.ts_model = Temperature_Scaling()
                self.ts_model.load_state_dict(torch.load(self.cfg.test.ts_path)['state_dict'])
                self.ts_model = self.ts_model.to(self.device)
            
            if self.cfg.test.ts_type == 'lts':
                                
                self.ts_model = LTS_CamVid_With_Image(input_channels=self.cfg.model.num_inp_channels, 
                                                    num_classes=self.cfg.model.num_classes)
                self.ts_model.load_state_dict(torch.load(self.cfg.test.ts_path)['state_dict'])
                self.ts_model = self.ts_model.to(self.device)
                
            if self.cfg.test.ts_type == 'grid':
                # self.temperature = self.find_best_temperature(self.val_loader)
                self.ts_model = self.find_best_temperature_optimize(self.val_loader)
                # if self.cfg.wandb.enable:
                #     wandb.log({'T':self.temperature})
        
        self.eval_epoch(self.test_loader, phase="Test",post_temp=self.cfg.test.post_temperature, ts_type = self.cfg.test.ts_type)
