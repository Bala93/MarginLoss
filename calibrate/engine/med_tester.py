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
from calibrate.net import ModelWithTemperature
# from calibrate.losses import LabelSmoothConstrainedLoss
from calibrate.evaluation import (
    AverageMeter, LossMeter, SegmentEvaluator, SegmentCalibrateEvaluator
)
from calibrate.utils import (
    load_train_checkpoint, load_checkpoint, save_checkpoint, round_dict
)
from calibrate.utils.file_io import mkdir
from calibrate.utils.torch_helper import entropy, to_numpy, get_lr
from .tester import Tester
from calibrate.utils.misc import bratspostprocess
from tqdm import tqdm

logger = logging.getLogger(__name__)


class MedSegmentTester(Tester):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)

    def build_meter(self):
        self.batch_time_meter = AverageMeter()
        self.num_classes = self.cfg.model.num_classes
        self.evaluator = SegmentEvaluator(
            self.test_loader.dataset.classes,
            ignore_index=255,
            ishd=True
        )
        self.calibrate_evaluator = SegmentCalibrateEvaluator(
            self.num_classes,
            num_bins=self.cfg.calibrate.num_bins,
            ignore_index=255,
            device=self.device
        )

    @torch.no_grad()
    def eval_epoch(self, data_loader, phase="Val"):
        self.reset_meter()
        self.model.eval()

        end = time.time()
        # fsave = open(osp.join(self.work_dir, "segment_{}_calibrate.txt".format(self.cfg.loss.name)), "w")
        for i, (inputs, labels) in enumerate(tqdm(data_loader)):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            # forward
            outputs = self.model(inputs)
            if isinstance(outputs, Dict):
                outputs = outputs["out"]
            # metric
            predicts = F.softmax(outputs, dim=1)
            pred_labels = torch.argmax(predicts, dim=1)
            
            ## to convert the brats to the metric setup
            if self.cfg.data.name == 'brain':
                bratspostprocess(pred_labels, labels)
            
            self.evaluator.update(
                to_numpy(pred_labels),
                to_numpy(labels)
            )
            self.calibrate_evaluator.update(
                outputs, labels
            )
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
        log_dict["samples"] = self.evaluator.num_samples()
        metric = self.evaluator.mean_score()
        log_dict.update(metric)
        calibrate_metric, calibrate_table_data = self.calibrate_evaluator.mean_score(isprint=False)
        log_dict.update(calibrate_metric)
        logger.info("{} Epoch\t{}".format(
            phase, json.dumps(round_dict(log_dict))
        ))
        class_table_data = self.evaluator.class_score(isprint=True, return_dataframe=True)
        
        class_hd_list = class_table_data['hd'].to_list()[:-1]
        class_dice_list = class_table_data['dsc'].to_list()[:-1]
        class_name_list = class_table_data['Class'].to_list()[:-1]
        
        for ii in range(len(class_name_list)):
            key = 'dsc-{}'.format(class_name_list[ii])
            val = class_dice_list[ii]
            log_dict.update({key:val})    
            
            key = 'hd-{}'.format(class_name_list[ii])
            val = class_hd_list[ii]
            log_dict.update({key:val})    
        
        logger.info("\n" + AsciiTable(calibrate_table_data).table)
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
                        columns=calibrate_table_data[0],
                        data=calibrate_table_data[1:]
                    )
                )
            wandb.log(wandb_log_dict)

    def test(self):
        logger.info(
            "Everything is perfect so far. Let's start testing. Good luck!"
        )
        self.eval_epoch(self.test_loader, phase="Test")