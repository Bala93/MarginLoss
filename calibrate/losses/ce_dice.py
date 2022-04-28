
import torch
import torch.nn as nn
import torch.nn.functional as F
from .dice import DiceLoss


class CEDiceLoss(nn.Module):
    def __init__(self):
        super(CEDiceLoss, self).__init__()
    
        self.ce = nn.CrossEntropyLoss()
        self.dice = DiceLoss()

    @property
    def names(self):
        return "loss", "loss_ce", "loss_dice"

    def forward(self, inputs, targets):
        
        celoss = self.ce(inputs, targets)

        diceloss = self.dice(inputs, targets)

        loss = celoss + diceloss

        return loss, celoss, diceloss