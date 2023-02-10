import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import get_svls_filter_2d

class AdaptMarginSVLS(nn.Module):
    """Add marginal penalty to logits:
        CE + alpha * max(0, max(l^n) - l^n - margin)
    """
    def __init__(self,
                 classes=None,
                 margin=10,
                 alpha=1.0,
                 ignore_index=-100,
                 sigma=1,
                 mu=0,
                 schedule="",
                 max_alpha=100.0,
                 step_size=100):
        
        super().__init__()
        assert schedule in ("", "add", "multiply", "step")
        
        self.margin = margin
        self.alpha = alpha
        self.ignore_index = ignore_index
        
        self.schedule = schedule
        self.max_alpha = max_alpha
        self.step_size = step_size

        self.nc = classes

        self.cross_entropy = nn.CrossEntropyLoss()

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
    
    def get_constr_target(self, mask, nc):
        
        
        mask = mask.unsqueeze(0) ## unfold works for 4d. 
        
        bs, _, h, w = mask.shape
        unfold = torch.nn.Unfold(kernel_size=(3, 3),padding=1)    
        umask = unfold(mask.float())
        
        rmask = []
        
        for ii in range(nc):
            rmask.append(torch.sum(umask == ii,1)/9)
            
        rmask = torch.stack(rmask,dim=1)
        rmask = rmask.reshape(bs, nc, h, w)

        return rmask
        

    def forward(self, inputs, targets):
        
        loss_ce = self.cross_entropy(inputs, targets)
        
        utargets = self.get_constr_target(targets,self.nc)
        loss_margin = F.relu(torch.abs(utargets-inputs)).mean()       

        loss = loss_ce + self.alpha * loss_margin

        return loss, loss_ce, loss_margin
