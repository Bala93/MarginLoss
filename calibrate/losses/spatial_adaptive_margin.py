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
                 kernel_size=3,
                 kernel_ops='mean',
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
        self.ks = kernel_size
        self.kernel_ops = kernel_ops
        self.cross_entropy = nn.CrossEntropyLoss()
        if kernel_ops == 'gaussian':
            self.svls_layer = get_svls_filter_2d(ksize=kernel_size, sigma=sigma, channels=classes)

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
    
    def get_constr_target(self, mask):
        
        mask = mask.unsqueeze(1) ## unfold works for 4d. 
        
        bs, _, h, w = mask.shape
        unfold = torch.nn.Unfold(kernel_size=(self.ks, self.ks),padding=self.ks // 2)    
        umask = unfold(mask.float())
        rmask = []
        
        if self.kernel_ops == 'mean':        
            
            for ii in range(self.nc):
                rmask.append(torch.sum(umask == ii,1)/self.ks**2)
                
        elif self.kernel_ops == 'max':
            for ii in range(self.nc):
                rmask.append((torch.max(umask,1)[0] == ii).int())
                        
        elif self.kernel_ops == 'min':
            for ii in range(self.nc):
                rmask.append((torch.min(umask,1)[0] == ii).int())
        
        elif self.kernel_ops == 'median':
            for ii in range(self.nc):
                rmask.append((torch.median(umask,1)[0] == ii).int())

        elif self.kernel_ops == 'mode':
            for ii in range(self.nc):
                rmask.append((torch.mode(umask,1)[0] == ii).int())
                
        if self.kernel_ops == 'gaussian':
            
            oh_labels = F.one_hot(mask[:,0].to(torch.int64), num_classes = self.nc).contiguous().permute(0,3,1,2).float()
            rmask = self.svls_layer(oh_labels)
            
            return rmask
                
        rmask = torch.stack(rmask,dim=1)
        rmask = rmask.reshape(bs, self.nc, h, w)
            
        return rmask
        

    def forward(self, inputs, targets):
        
        loss_ce = self.cross_entropy(inputs, targets)
        
        utargets = self.get_constr_target(targets)
        
        loss_margin = torch.abs(utargets-inputs).mean()       

        loss = loss_ce + self.alpha * loss_margin

        return loss, loss_ce, loss_margin
