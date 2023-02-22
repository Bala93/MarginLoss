from medpy.metric import binary
from calibrate.evaluation import metrics
import numpy as np
import math
import matplotlib.pyplot as plt
import torch.nn.functional as F
import seaborn as sns

plt.rcParams.update({'font.size': 20})
plt.style.use('ggplot')

# Some keys used for the following dictionaries
COUNT = 'count'
CONF = 'conf'
ACC = 'acc'
BIN_ACC = 'bin_acc'
BIN_CONF = 'bin_conf'


# def shape_metrics(pred_label, label, num_classes):

#     hd = []
#     dsc = []

#     for cls in range(1,num_classes):
        
#         try:
#             pred_bw = pred_label == cls
#             label_bw = label == cls
            
#             dsc.append(binary.dc(pred_bw,label_bw))
#             hd.append(binary.asd(pred_bw, label_bw))
            
#         except Exception as e:
#             print (e)
#             hd.append(0)
#             dsc.append(1)
            
#     return np.round(np.mean(dsc),4), np.round(np.mean(hd),4)

def shape_metrics(pred_label, label, num_classes, fpath, is_hd=False):

    hd = []
    dsc = []
    class_nums = np.unique(label)[1:]

    for cls in class_nums:
        
        try:
            pred_bw = pred_label == cls
            label_bw = label == cls
            if is_hd:
                hd.append(binary.asd(pred_bw, label_bw))
            else:
                hd.append(0)
            # hd.append(binary.hd(pred_bw, label_bw))
            dsc.append(binary.dc(pred_bw,label_bw))
            
        except Exception as e:
            print (fpath, cls)
            hd.append(0)
            dsc.append(0)
            #dsc.append(binary.dc(pred_bw,label_bw))
            
    return np.round(np.mean(dsc),4), np.round(np.mean(hd),4)


def metric_dsc_hd(gt, pred,cno):
    
    try:
        dsc_m = np.round(binary.dc(pred == cno, gt == cno),4)
        hd_m = np.round(binary.asd(pred == cno, gt == cno),4)
        
    except Exception as e:
        # print (e)
        hd_m = 0
        dsc_m = 1
        
    return dsc_m, hd_m
    

def calib_metrics(preds, targets):
    
    ece = metrics.ECELoss()(preds, targets).item()
    cece = metrics.ClasswiseECELoss()(preds, targets).item()
    aece = metrics.AdaptiveECELoss()(preds, targets).item()
    
    nll = F.cross_entropy(preds, targets.long()).item()
    
    return np.round(ece,4), np.round(cece,4), np.round(aece,4), np.round(nll,4)

def metric_multi_class(gt, pred, ignore_bg=True, is_hd=False):
    
    try:
        cnums1 = np.unique(gt)
        cnums2 = np.unique(pred)
        cnums = np.union1d(cnums1, cnums2)
        
        if ignore_bg:
            cnums = cnums[cnums!=0] 

        dsc_cls = []
        hd_cls = []

        for cno in cnums:

            dsc_cls.append(round(binary.dc(pred == cno, gt == cno),4))
            if is_hd:
                hd_cls.append(round(binary.hd(pred == cno , gt == cno),4))
            else:
                hd_cls.append(0)
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
            # bin_dict[binn][BIN_ACC] = 0
            # bin_dict[binn][BIN_CONF] = 0
            ## for getting a 45d line.
            bin_dict[binn][BIN_ACC] = binn / num_bins
            bin_dict[binn][BIN_CONF] = binn / num_bins

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
    # plt.figure(figsize=(10, 8))  # width:20, height:3
    plt.plot(bns, bns, color='pink', label='Expected')
    plt.plot(bns, y, color='blue', label='Actual')
    plt.ylabel('Accuracy')
    plt.xlabel('Confidence')
    plt.legend()
    plt.title(title)
    # plt.show()
    
    return bns, y


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
    # plt.figure(figsize=(10, 8))  # width:20, height:3
    plt.bar(bns, y, align='edge', width=0.05,
            color='blue', alpha=0.5, label='Percentage samples')
    plt.ylabel('Percentage of samples')
    plt.xlabel('Confidence')
    plt.title(title)
    plt.show()
    
def reliability_curve_obj(obj, confs, preds, labels, title=None, num_bins=15):
    '''
    Method to draw a reliability plot from a model's predictions and confidences.
    '''
    bin_dict = _populate_bins(confs, preds, labels, num_bins)
    bns = [(i / float(num_bins)) for i in range(num_bins)]
    y = []
    for i in range(num_bins):
        y.append(bin_dict[i][BIN_ACC])
    # plt.figure(figsize=(10, 8))  # width:20, height:3
    obj.plot(bns, bns, color='pink', label='Expected')
    obj.plot(bns, y, color='blue', label='Actual')
    obj.set_ylabel('Accuracy')
    obj.set_xlabel('Confidence')
    obj.legend()
    obj.set_title(title)
    # plt.show()
    
def expected_calibration_error(confs, preds, labels, num_bins=15):
    bin_dict = _populate_bins(confs, preds, labels, num_bins)
    num_samples = len(labels)
    ece = 0
    for i in range(num_bins):
        bin_accuracy = bin_dict[i][BIN_ACC]
        bin_confidence = bin_dict[i][BIN_CONF]
        bin_count = bin_dict[i][COUNT]
        ece += (float(bin_count) / num_samples) * \
            abs(bin_accuracy - bin_confidence)
    return round(ece,4)


def reliability_curve_save(confs, preds, labels, metric, savepath, num_bins=15):
    '''
    Method to draw a reliability plot from a model's predictions and confidences.
    '''
    bin_dict = _populate_bins(confs, preds, labels, num_bins)
    bns = [(i / float(num_bins)) for i in range(num_bins)]
    y = []
    for i in range(num_bins):
        y.append(bin_dict[i][BIN_ACC])
    plt.figure(figsize=(10, 8))  # width:20, height:3
    plt.plot(bns, bns, color='pink', label='Expected',linewidth=5)
    plt.plot(bns, y, color='blue', label='Actual',linewidth=5)
    plt.text(0.9, 0.1, metric, size=50, ha="right", va="bottom",bbox=dict(boxstyle="square",ec=(1., 0.5, 0.5),fc=(1., 0.8, 0.8),))
    plt.ylabel('Accuracy',fontsize=32)
    plt.xlabel('Confidence',fontsize=32)
    plt.legend(fontsize=32,loc='upper left')
    plt.savefig(savepath,bbox_inches='tight')
    # plt.show()


def bratspostprocess(outputs, labels):
    
    def process(mask):
        nmask = np.zeros(shape=mask.shape, dtype=mask.dtype)
        nmask[mask > 0 ] = 3
        nmask[np.logical_or(mask == 2 , mask == 3)] = 1
        nmask[mask == 2] = 2
        return nmask 
    
    noutputs = process(outputs)
    nlabels = process(labels)

    return noutputs, nlabels

def reliability_curve_density(confs, preds, labels, title=None, num_bins=15):
    '''
    Method to draw a reliability plot from a model's predictions and confidences.
    '''
    bin_dict = _populate_bins(confs, preds, labels, num_bins)
    bns = [(i / float(num_bins)) for i in range(num_bins)]
    y = []
    for i in range(num_bins):
        y.append(bin_dict[i][BIN_ACC])
    # plt.figure(figsize=(10, 8))  # width:20, height:3
    # plt.plot(bns, bns, color='pink', label='Expected')
    # plt.plot(bns, y, color='blue', label='Actual')
    sns.jointplot(bns,y,kind='kde')
    # plt.hist2d(bns, y, bins=15, cmap='Blues')
    plt.ylabel('Accuracy')
    plt.xlabel('Confidence')
    plt.legend()
    plt.title(title)
    # plt.show()
