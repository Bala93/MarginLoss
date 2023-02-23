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

logger = logging.getLogger(__name__)


def shape_metrics(pred_label, label, num_classes, is_hd=False):

    hd = []
    dsc = []
    # class_nums = np.unique(label)[1:]

    for cls in range(num_classes):
        
        try:
            pred_bw = pred_label == cls
            label_bw = label == cls
            if is_hd:
                hd.append(binary.asd(pred_bw, label_bw))
            else:
                hd.append(0)
            # hd.append(binary.hd(pred_bw, label_bw))
            dsc.append(np.round(binary.dc(pred_bw,label_bw),4))
            
        except Exception as e:
            print ("Empty volume", cls)
            hd.append(0)
            dsc.append(0)
            #dsc.append(binary.dc(pred_bw,label_bw))
            
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
                 ishd = False) -> None:
        super().__init__()
        self.classes = classes
        self.num_classes = len(self.classes)
        self.ignore_index = ignore_index
        self.ishd = ishd

    def num_samples(self):
        return self.nsamples

    def reset(self):
        self.total_dsc = np.empty((0,self.num_classes), dtype=np.float32)
        self.total_hd = np.empty((0,self.num_classes), dtype=np.float32)
        self.total_ece = np.empty((0,1), dtype=np.float32)
        self.total_cece = np.empty((0,1), dtype=np.float32)

    def main_metric(self):
        return "miou"

    def ignore_background(self, pred: np.ndarray, target: np.ndarray):
        pred = pred[:, 1:] if pred.shape[1] > 1 else pred
        target = target[:, 1:] if target.shape[1] > 1 else target
        return pred, target

    def update(self, pred: np.ndarray, target: np.ndarray, logits, labels):
        """Update all the metric from batch size prediction and target.

        Args:
            pred: predictions to be evaluated in one-hot formation
            y: ground truth. It should be one-hot format.
        """

        n = pred.shape[0]
                
        dsc, hd = shape_metrics(pred, target,self.num_classes)
        
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
