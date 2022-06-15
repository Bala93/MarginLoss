import logging
import numpy as np
import os
import random
from datetime import datetime
import torch
from copy import deepcopy
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def set_random_seed(seed: int = None, deterministic: bool = False):
    """
    Set the random seed for the RNG in torch, numpy and python.

    Args:
        seed (int): if None, will use a strong random seed.
        deterministic (bool):  Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
    """
    if seed is None:
        seed = (
            os.getpid()
            + int(datetime.now().strftime("%S%f"))
            + int.from_bytes(os.urandom(2), "big")
        )
        logger = logging.getLogger(__name__)
        logger.info("Using a generated random seed {}".format(seed))
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_logfile(logger):
    if len(logger.root.handlers) == 1:
        return None
    else:
        return logger.root.handlers[1].baseFilename


def round_dict(d, decimals=5):
    """
    Return a new dictionary with all the flating values rounded
    with the sepcified number of decimals
    """
    ret = deepcopy(d)
    for key in ret:
        if isinstance(ret[key], float):
            ret[key] = round(ret[key], decimals)
    return ret

def bratspostprocess(outputs, labels):
    
    def process(mask):
        nmask = torch.zeros(size=mask.shape, dtype=mask.dtype)
        nmask[mask > 0 ] = 3
        nmask[torch.logical_or(mask == 2 , mask == 3)] = 1
        nmask[mask == 2] = 2
        return nmask 
    
    noutputs = process(outputs)
    nlabels = process(labels)

    return noutputs, nlabels


class BND(nn.Module):

    def __init__(self, device):
        super(BND,self).__init__()

        # HSOBEL_WEIGHTS = np.array([[1, 2, 1],[0, 0, 0],[-1, -2, -1]]) / 4.0
        HSOBEL_WEIGHTS = np.array([[0, 1, 0],[0, 0, 0],[0, -1, 0]])
        HSOBEL_WEIGHTS = HSOBEL_WEIGHTS.astype(np.float64)
        VSOBEL_WEIGHTS = HSOBEL_WEIGHTS.T

        HSOBEL_WEIGHTS = torch.from_numpy(HSOBEL_WEIGHTS)
        VSOBEL_WEIGHTS = torch.from_numpy(VSOBEL_WEIGHTS)

        HSOBEL_WEIGHTS = HSOBEL_WEIGHTS.to(device)
        VSOBEL_WEIGHTS = VSOBEL_WEIGHTS.to(device)

        self.HSOBEL_WEIGHTS = HSOBEL_WEIGHTS.unsqueeze(0).unsqueeze(0)
        self.VSOBEL_WEIGHTS = VSOBEL_WEIGHTS.unsqueeze(0).unsqueeze(0)

    def forward(self,img):
        
        img = img.unsqueeze(1).double()
        edge_torch_H = F.conv2d(img,self.HSOBEL_WEIGHTS,padding=1)
        edge_torch_V = F.conv2d(img,self.VSOBEL_WEIGHTS,padding=1)
        edge_abs = torch.sqrt(edge_torch_H **2 + edge_torch_V **2 )
        
        edge = edge_abs > 0
        
        return edge