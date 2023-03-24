import logging
import numpy as np
from terminaltables import AsciiTable
import pandas as pd
from typing import List, Optional
import wandb
from medpy.metric import binary
import torch.nn.functional as F
from .evaluator import DatasetEvaluator
from calibrate.evaluation import metrics
import torch
import os 

logger = logging.getLogger(__name__)


def brats_postprocess(mask1, mask2, cno):

    nmask1 = np.zeros(shape=mask1.shape)
    nmask2 = np.zeros(shape=mask2.shape)

    if cno == 0: 
        idx = mask1 == 0
        nmask1[idx] = 1
        idx = mask2 == 0
        nmask2[idx] = 1
    
    if cno == 1: # ET
        idx = mask1 == 3
        nmask1[idx] = 1
        idx = mask2 == 3
        nmask2[idx] = 1


    if cno == 2: # TC
        idx = np.logical_or(mask1 == 1, mask1 == 3)
        nmask1[idx] = 2
        idx = np.logical_or(mask2 == 1, mask2 == 3)
        nmask2[idx] = 2


    if cno == 3: # WT
        idx = mask1 > 0
        nmask1[idx] = 3
        idx = mask2 > 0
        nmask2[idx] = 3

    return nmask1, nmask2


def shape_metrics(pred_label, label, num_classes, is_hd=False, dataset_type=None):

    hd = []
    dsc = []
    # class_nums = np.unique(label)[1:]

    for cls in range(num_classes):
        
        try:
            
            if dataset_type == 'brain19' :
                pred_label_, label_ = brats_postprocess(pred_label, label, cls)
                pred_bw = pred_label_ == cls
                label_bw = label_ == cls
                
            else:
                pred_label_, label_ = pred_label, label                
                pred_bw = pred_label_ == cls
                label_bw = label_ == cls
            
            if np.sum(label_bw) == 0:
                hd.append(0)
                dsc.append(1)
            
            elif is_hd:
                hd.append(binary.hd95(pred_bw, label_bw))
                dsc.append(np.round(binary.dc(pred_bw,label_bw),4))
                
            else:
                hd.append(0)
                dsc.append(np.round(binary.dc(pred_bw,label_bw),4))
                
            
        except Exception as e:
            print ("Empty volume", cls, e)
            hd.append(0)
            # dsc.append(1)
            dsc.append(binary.dc(pred_bw,label_bw))
            
    # return np.round(np.mean(dsc),4), np.round(np.mean(hd),4)
    return dsc, hd

def calib_metrics(preds, targets):
    
    ece = metrics.ECELoss()(preds, targets).item()
    cece = metrics.ClasswiseECELoss()(preds, targets).item()
    # aece = metrics.AdaptiveECELoss()(preds, targets).item()
    
    # nll = F.cross_entropy(preds, targets.long()).item()
    
    return np.round(ece,4), np.round(cece,4), #np.round(aece,4), np.round(nll,4)


class CalibSegmentEvaluator(DatasetEvaluator):
    def __init__(self,
                 classes: Optional[List[str]] = None,
                 ignore_index: int = -1,
                 ishd = False, dataset_type=None) -> None:
        super().__init__()
        self.classes = classes
        self.num_classes = len(self.classes)
        self.ignore_index = ignore_index
        self.ishd = ishd
        self.dataset_type = dataset_type

    def num_samples(self):
        return self.nsamples

    def reset(self):
        self.total_dsc = np.empty((0,self.num_classes), dtype=np.float32)
        self.total_hd = np.empty((0,self.num_classes), dtype=np.float32)
        self.total_ece = np.empty((0,1), dtype=np.float32)
        self.total_cece = np.empty((0,1), dtype=np.float32)
        self.file_names = []

    def main_metric(self):
        return "miou"

    def ignore_background(self, pred: np.ndarray, target: np.ndarray):
        pred = pred[:, 1:] if pred.shape[1] > 1 else pred
        target = target[:, 1:] if target.shape[1] > 1 else target
        return pred, target

    def update(self, pred: np.ndarray, target: np.ndarray, logits, labels, fpath):
        """Update all the metric from batch size prediction and target.

        Args:
            pred: predictions to be evaluated in one-hot formation
            y: ground truth. It should be one-hot format.
        """

        n = pred.shape[0]
                
        dsc, hd = shape_metrics(pred, target,self.num_classes,self.ishd, self.dataset_type)
        
        # print (dsc)
        
        n, c, x, y = logits.shape
        
        labels = labels.reshape(n * x * y)
        logits = torch.einsum("ncxy->nxyc", logits)
        logits = logits.reshape(n * x * y, -1)
        index = torch.nonzero(labels != 0).squeeze()
        
        logits = logits[index, :]
        labels = labels[index]
        
        ece, cece  = calib_metrics(logits, labels)
                
        # update the total
        self.total_dsc = np.vstack([self.total_dsc, dsc])
        self.total_hd = np.vstack([self.total_hd, hd])
        self.total_ece = np.vstack([self.total_ece, ece])
        self.total_cece = np.vstack([self.total_cece, cece])
        self.file_names.append(os.path.basename(fpath[0]))
        
    def mean_score(self):
        
        mdsc = self.total_dsc[:,1:].mean()
        mhd = self.total_hd[:,1:].mean()
        mece = self.total_ece.mean()
        mcece = self.total_cece.mean()
        
        return {"mdsc": mdsc, "mhd": mhd, "mece": mece, "mcece": mcece}

    def class_score(self, isprint=True, return_dataframe=False):
                
        class_dice = self.total_dsc[:,1:].mean(axis=0)
        class_hd = self.total_hd[:,1:].mean(axis=0)
        
        columns = ["id", "Class", "dsc", "hd"]

        class_table_data = [columns]
        for i in range(class_dice.shape[0]):
            class_table_data.append(
                [i] + [self.classes[i + 1]]
                + ["{:.4f}".format(class_dice[i])]
                + ["{:.4f}".format(class_hd[i])]
            )
        class_table_data.append(
            [""] + ["mean"]
            + ["{:.4f}".format(np.mean(class_dice))]
            + ["{:.4f}".format(np.mean(class_hd))]
        )

        if isprint:
            table = AsciiTable(class_table_data)
            logger.info("\n" + table.table)
            
        if return_dataframe:
            data = {key: [] for key in columns}
            for i in range(class_dice.shape[0]):
                data[columns[0]].append(i)
                data[columns[1]].append(self.classes[i + 1])
                data[columns[2]].append(class_dice[i])
                data[columns[3]].append(class_hd[i])
                
            data[columns[0]].append(None)
            data[columns[1]].append("mean")
            data[columns[2]].append(np.mean(class_dice))
            data[columns[3]].append(np.mean(class_hd))

            return pd.DataFrame(data, columns=columns)
        
    def calib_score(self, isprint=False, return_dataframe=True):
        
        ece = self.total_ece.mean()
        cece = self.total_cece.mean()

        columns = ["ece", "cece"]
        table_data = [columns]
        table_data.append(
            [
                "{:.5f}".format(ece),
                "{:.5f}".format(cece),
            ]
        )

        if isprint:
            table = AsciiTable(table_data)
            logger.info("\n" + table.table)
            
        if return_dataframe:
            data = {key: [] for key in columns}
            data[columns[0]].append(ece)
            data[columns[1]].append(cece)
            
            return pd.DataFrame(data, columns=columns)
        

    def wandb_score_table(self):
        table_data = self.class_score(isprint=False, return_dataframe=True)
        return wandb.Table(dataframe=table_data)

    def save_csv(self,save_dir):
        
        with open(os.path.join(save_dir,'metrics.csv'),'w') as f:
            for fname, dsc in zip(self.file_names, self.total_dsc):
                f.write(','.join([fname] + [str(ii) for ii in dsc]) + '\n')