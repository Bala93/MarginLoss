from medpy.metric import binary
from calibrate.evaluation import metrics
import numpy as np

def shape_metrics(pred_label, label, num_classes):

    hd = []
    dsc = []

    for cls in range(num_classes):
        pred_bw = pred_label == cls
        label_bw = label == cls
        if np.sum(pred_bw) & np.sum(label_bw):
            hd.append(binary.hd(pred_bw, label_bw))
            dsc.append(binary.dc(pred_bw,label_bw))
        else:
            hd.append(0)
            dsc.append(1)
            
    return np.round(np.mean(dsc),4), np.round(np.mean(hd),2)

def calib_metrics(preds, targets):
    
    ece = metrics.ECELoss()(preds, targets).item()
    cece = metrics.ClasswiseECELoss()(preds, targets).item()
     
    return np.round(ece,4), np.round(cece,4)
