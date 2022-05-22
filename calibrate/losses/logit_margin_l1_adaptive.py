import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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


class LogitMarginBondary(nn.Module):
    """Add marginal penalty to logits:
        CE + alpha * max(0, max(l^n) - l^n - margin)
    """
    def __init__(self,
                 margin=10,
                 alpha=0.1,
                 ignore_index=-100,
                 mu=0,
                 schedule="",
                 max_alpha=100.0,
                 step_size=100,
                 device='cuda:0'):
        
        super().__init__()
        assert schedule in ("", "add", "multiply", "step")
        self.margin = margin
        self.alpha = alpha
        self.ignore_index = ignore_index
        self.mu = mu
        self.schedule = schedule
        self.max_alpha = max_alpha
        self.step_size = step_size

        self.cross_entropy = nn.CrossEntropyLoss()
        self.bnd = BND(device)
        

    @property
    def names(self):
        return "loss", "loss_ce", "loss_margin_l1"

    def schedule_alpha(self, epoch):
        if self.schedule == "add":
            self.alpha = min(self.alpha + self.mu, self.max_alpha)
        elif self.schedule == "multiply":
            self.alpha = min(self.alpha * self.mu, self.max_alpha)
        elif self.schedule == "step":
            if (epoch + 1) % self.step_size == 0:
                self.alpha = min(self.alpha * self.mu, self.max_alpha)

    def get_diff(self, inputs):
        max_values = inputs.max(dim=1)
        max_values = max_values.values.unsqueeze(dim=1).repeat(1, inputs.shape[1])
        diff = max_values - inputs
        return diff

    def forward(self, inputs, targets):
        
        bndry = self.bnd(targets)
        
        if inputs.dim() > 2:
            inputs = inputs.view(inputs.size(0), inputs.size(1), -1)  # N,C,H,W => N,C,H*W
            inputs = inputs.transpose(1, 2)    # N,C,H*W => N,H*W,C
            inputs = inputs.contiguous().view(-1, inputs.size(2))   # N,H*W,C => N*H*W,C
            targets = targets.view(-1)
            bndry = bndry.view(-1)

        if self.ignore_index >= 0:
            index = torch.nonzero(targets != self.ignore_index).squeeze()
            inputs = inputs[index, :]
            targets = targets[index]

        loss_ce = self.cross_entropy(inputs, targets)

        diff = self.get_diff(inputs)
        # loss_margin = torch.clamp(diff - self.margin, min=0).mean()
        
        margin_adaptive = torch.ones(diff.shape,device=diff.device)
        margin_adaptive[bndry] = 0
        margin_adaptive = margin_adaptive * self.margin
        
        loss_margin = F.relu(diff-margin_adaptive).mean()

        loss = loss_ce + self.alpha * loss_margin

        return loss, loss_ce, loss_margin
