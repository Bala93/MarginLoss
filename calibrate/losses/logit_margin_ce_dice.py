from xml.dom.expatbuilder import InternalSubsetExtractor
import torch
import torch.nn as nn
import torch.nn.functional as F

class LogitMarginDICEL1(nn.Module):
    """Add marginal penalty to logits:
        CE + alpha * max(0, max(l^n) - l^n - margin)
    """
    def __init__(self,
                 margin=10,
                 alpha=1.0,
                 ignore_index=-100,
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

        self.cross_entropy = nn.CrossEntropyLoss()

    @property
    def names(self):
        return "loss", "loss_ce", "loss_margin_l1", "loss_dice"

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

    def get_dice(self, inputs, targets, eps = 1e-5):

        numclasses = inputs.shape[1]
        targets = targets.unsqueeze(1)

        sh = list(targets.shape)
        sh[1] = numclasses
        
        o = torch.zeros(size=sh,dtype=inputs.dtype,device=targets.device)
        labels = o.scatter_(1,index=targets,value=1)

        reduce_axis = torch.arange(2, len(inputs.shape)).tolist()

        num = torch.sum(labels * inputs, reduce_axis)
        den = torch.sum(labels, reduce_axis) + torch.sum(inputs, reduce_axis)
        loss = (2 * num + eps) / (den + eps)

        loss = torch.mean(loss)

        return 1 - loss

    def forward(self, inputs, targets):
        if inputs.dim() > 2:
            inputs = inputs.view(inputs.size(0), inputs.size(1), -1)  # N,C,H,W => N,C,H*W
            inputs = inputs.transpose(1, 2)    # N,C,H*W => N,H*W,C
            inputs = inputs.contiguous().view(-1, inputs.size(2))   # N,H*W,C => N*H*W,C
            targets = targets.view(-1)

        if self.ignore_index >= 0:
            index = torch.nonzero(targets != self.ignore_index).squeeze()
            inputs = inputs[index, :]
            targets = targets[index]

        loss_ce = self.cross_entropy(inputs, targets)

        diff = self.get_diff(inputs)
        # loss_margin = torch.clamp(diff - self.margin, min=0).mean()
        loss_margin = F.relu(diff-self.margin).mean()

        loss_dice = self.get_dice(inputs, targets)

        loss = loss_ce + loss_dice + self.alpha * loss_margin

        return loss, loss_ce, loss_margin, loss_dice
