import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import get_svls_filter_2d

class LogitMarginSVLSL1(nn.Module):
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
        self.mu = mu
        self.schedule = schedule
        self.max_alpha = max_alpha
        self.step_size = step_size

        self.cls = torch.tensor(classes)
        self.cls_idx = torch.arange(self.cls).reshape(1, self.cls).cuda()

        self.svls_layer, self.svls_kernel = get_svls_filter_2d(sigma=sigma, channels=classes)
        self.svls_kernel = self.svls_kernel.cuda()


        # self.cross_entropy = nn.CrossEntropyLoss()

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

        oh_labels = (targets[...,None] == self.cls_idx).permute(0,3,1,2)
        oh_labels = F.pad(oh_labels.float(), (1,1,1,1), mode='replicate')

        svls_labels = self.svls_layer(oh_labels) / self.svls_kernel.sum()

        loss_ce = (-svls_labels * F.log_softmax(inputs, dim=1)).sum(dim=1).mean()

        if inputs.dim() > 2:
            inputs = inputs.view(inputs.size(0), inputs.size(1), -1)  # N,C,H,W => N,C,H*W
            inputs = inputs.transpose(1, 2)    # N,C,H*W => N,H*W,C
            inputs = inputs.contiguous().view(-1, inputs.size(2))   # N,H*W,C => N*H*W,C
            targets = targets.view(-1)

        if self.ignore_index >= 0:
            index = torch.nonzero(targets != self.ignore_index).squeeze()
            inputs = inputs[index, :]
            targets = targets[index]

        diff = self.get_diff(inputs)
        # loss_margin = torch.clamp(diff - self.margin, min=0).mean()
        loss_margin = F.relu(diff-self.margin).mean()       

        loss = loss_ce + self.alpha * loss_margin

        return loss, loss_ce, loss_margin
