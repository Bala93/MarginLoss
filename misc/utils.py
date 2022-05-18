from medpy.metric import binary
from calibrate.evaluation import metrics
import numpy as np

import math
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 20})

# Some keys used for the following dictionaries
COUNT = 'count'
CONF = 'conf'
ACC = 'acc'
BIN_ACC = 'bin_acc'
BIN_CONF = 'bin_conf'


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


def metric_multi_class(gt, pred):
    
    try:
        cnums = np.unique(gt)

        dsc_cls = []
        hd_cls = []

        for cno in cnums:

            dsc_cls.append(round(binary.dc(pred == cno, gt == cno),4))
            hd_cls.append(round(binary.hd(pred == cno , gt == cno),4))
        dsc_m, hd_m = np.round(np.mean(dsc_cls),4), np.round(np.mean(hd_cls),4)
        
    except:
        
        dsc_m, hd_m = 1, 0
        
    return dsc_m, hd_m
    
    

def _bin_initializer(bin_dict, num_bins=10):
    for i in range(num_bins):
        bin_dict[i][COUNT] = 0
        bin_dict[i][CONF] = 0
        bin_dict[i][ACC] = 0
        bin_dict[i][BIN_ACC] = 0
        bin_dict[i][BIN_CONF] = 0


def _populate_bins(confs, preds, labels, num_bins=10):
    bin_dict = {}
    for i in range(num_bins):
        bin_dict[i] = {}
    _bin_initializer(bin_dict, num_bins)
    num_test_samples = len(confs)

    for i in range(0, num_test_samples):
        confidence = confs[i]
        prediction = preds[i]
        label = labels[i]
        binn = int(math.ceil(((num_bins * confidence) - 1)))
        bin_dict[binn][COUNT] = bin_dict[binn][COUNT] + 1
        bin_dict[binn][CONF] = bin_dict[binn][CONF] + confidence
        bin_dict[binn][ACC] = bin_dict[binn][ACC] + \
            (1 if (label == prediction) else 0)

    for binn in range(0, num_bins):
        if (bin_dict[binn][COUNT] == 0):
            bin_dict[binn][BIN_ACC] = 0
            bin_dict[binn][BIN_CONF] = 0
        else:
            bin_dict[binn][BIN_ACC] = float(
                bin_dict[binn][ACC]) / bin_dict[binn][COUNT]
            bin_dict[binn][BIN_CONF] = bin_dict[binn][CONF] / \
                float(bin_dict[binn][COUNT])
    return bin_dict


def reliability_plot(confs, preds, labels, title=None, num_bins=15):
    '''
    Method to draw a reliability plot from a model's predictions and confidences.
    '''
    bin_dict = _populate_bins(confs, preds, labels, num_bins)
    bns = [(i / float(num_bins)) for i in range(num_bins)]
    y = []
    for i in range(num_bins):
        y.append(bin_dict[i][BIN_ACC])
    # plt.figure(figsize=(10, 8))  # width:20, height:3
    plt.bar(bns, bns, align='edge', width=0.05, color='pink', label='Expected')
    plt.bar(bns, y, align='edge', width=0.05,
            color='blue', alpha=0.5, label='Actual')
    plt.ylabel('Accuracy')
    plt.xlabel('Confidence')
    plt.legend()
    # plt.title(title)
    # plt.show()
    
def reliability_curve(confs, preds, labels, title=None, num_bins=15):
    '''
    Method to draw a reliability plot from a model's predictions and confidences.
    '''
    bin_dict = _populate_bins(confs, preds, labels, num_bins)
    bns = [(i / float(num_bins)) for i in range(num_bins)]
    y = []
    for i in range(num_bins):
        y.append(bin_dict[i][BIN_ACC])
    plt.figure(figsize=(10, 8))  # width:20, height:3
    plt.plot(bns, bns, color='pink', label='Expected')
    plt.plot(bns, y, color='blue', label='Actual')
    plt.ylabel('Accuracy')
    plt.xlabel('Confidence')
    plt.legend()
    plt.title(title)
    plt.show()


def bin_strength_plot(confs, preds, labels, title=None,num_bins=15):
    '''
    Method to draw a plot for the number of samples in each confidence bin.
    '''
    bin_dict = _populate_bins(confs, preds, labels, num_bins)
    bns = [(i / float(num_bins)) for i in range(num_bins)]
    num_samples = len(labels)
    y = []
    for i in range(num_bins):
        n = (bin_dict[i][COUNT] / float(num_samples)) * 100
        y.append(n)
    plt.figure(figsize=(10, 8))  # width:20, height:3
    plt.bar(bns, y, align='edge', width=0.05,
            color='blue', alpha=0.5, label='Percentage samples')
    plt.ylabel('Percentage of samples')
    plt.xlabel('Confidence')
    plt.title(title)
    plt.show()