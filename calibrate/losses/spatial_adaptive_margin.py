import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import get_gaussian_kernel_2d, get_svls_filter_2d

class AdaptMarginSVLS(nn.Module):
    """Add marginal penalty to logits:
        CE + alpha * max(0, max(l^n) - l^n - margin)
    """
    def __init__(self,
                 classes=None,
                 kernel_size=3,
                 kernel_ops='mean',
                 margin=3,
                 alpha=0.1,
                 beta=0,
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
        self.beta = beta
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
        if kernel_ops == 'bilateral':
            self.gkernel = get_gaussian_kernel_2d(ksize=kernel_size, sigma=sigma)

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
    
    def get_constr_target(self, mask, img):
        
        mask = mask.unsqueeze(1) ## unfold works for 4d. 
        
        bs, _, h, w = mask.shape
        unfold = torch.nn.Unfold(kernel_size=(self.ks, self.ks),padding=self.ks // 2)    
        
        rmask = []
        
        if self.kernel_ops == 'mean':        
            umask = unfold(mask.float())
                
            for ii in range(self.nc):
                rmask.append(torch.sum(umask == ii,1)/self.ks**2)
                
        elif self.kernel_ops == 'max':
            umask = unfold(mask.float())
            for ii in range(self.nc):
                rmask.append((torch.max(umask,1)[0] == ii).int())
                        
        elif self.kernel_ops == 'min':
            umask = unfold(mask.float())
            for ii in range(self.nc):
                rmask.append((torch.min(umask,1)[0] == ii).int())
        
        elif self.kernel_ops == 'median':
            umask = unfold(mask.float())
            for ii in range(self.nc):
                rmask.append((torch.median(umask,1)[0] == ii).int())

        elif self.kernel_ops == 'mode':
            umask = unfold(mask.float())
            for ii in range(self.nc):
                rmask.append((torch.mode(umask,1)[0] == ii).int())
                
        if self.kernel_ops == 'gaussian':
            
            oh_labels = F.one_hot(mask[:,0].to(torch.int64), num_classes = self.nc).contiguous().permute(0,3,1,2).float()
            rmask = self.svls_layer(oh_labels)
            
            return rmask
        
        if self.kernel_ops == 'bilateral':
            
            umask = unfold(mask.float())
            uimg = unfold(img) # bs, 9, N
            cuimg = uimg[:,self.ks ** 2 // 2,:].unsqueeze(1) # bs, 1, N
            ugrad = torch.abs(uimg - cuimg) 
            ukernel = ugrad * self.gkernel.reshape(1,self.ks ** 2,1).to(ugrad.device)
            ohumask = F.one_hot(umask.to(torch.int64), num_classes = self.nc)
            ukernel = ukernel.unsqueeze(-1)
            rmask = ohumask  * ukernel
            rmask = torch.mean(rmask, dim=1)
            rmask = rmask.reshape(bs, self.nc, h, w)
            
            return rmask 
            
        rmask = torch.stack(rmask,dim=1)
        rmask = rmask.reshape(bs, self.nc, h, w)
            
        return rmask
        

    def forward(self, inputs, targets, imgs):
        
        loss_ce = self.cross_entropy(inputs, targets)
        
        utargets = self.get_constr_target(targets, imgs)
        
        loss_conf = torch.abs(utargets-inputs).mean()       
   
        if inputs.dim() > 2:
            inputs = inputs.view(inputs.size(0), inputs.size(1), -1)  # N,C,H,W => N,C,H*W
            inputs = inputs.transpose(1, 2)    # N,C,H*W => N,H*W,C
            inputs = inputs.contiguous().view(-1, inputs.size(2))   # N,H*W,C => N*H*W,C
            targets = targets.view(-1)     
            
        diff = self.get_diff(inputs)
        loss_margin = F.relu(diff-self.margin).mean()

        loss = loss_ce + self.alpha * loss_conf + self.beta * loss_margin

        return loss, loss_ce, loss_conf
