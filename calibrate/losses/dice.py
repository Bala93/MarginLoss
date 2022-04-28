
import torch
import torch.nn as nn
import torch.nn.functional as F
import monai


class DiceLoss(nn.Module):
    def __init__(self, include_background=False, to_onehot_y = True, softmax = True):

        super(DiceLoss, self).__init__()

        self.monai_dice = monai.losses.DiceLoss(to_onehot_y = True, 
                                                softmax = True)


    def forward(self, inputs, targets):

        targets = targets.unsqueeze(1)

        loss = self.monai_dice(inputs, targets)

        return loss
