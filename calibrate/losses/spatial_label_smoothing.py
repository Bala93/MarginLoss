import torch
import torch.nn.functional as F
from .utils import get_svls_filter_2d


class CELossWithSVLS(torch.nn.Module):
    def __init__(self, classes=None, sigma=1):
        super(CELossWithSVLS, self).__init__()
        self.cls = torch.tensor(classes)
        self.cls_idx = torch.arange(self.cls).reshape(1, self.cls).cuda()
        self.svls_layer, self.svls_kernel = get_svls_filter_2d(sigma=sigma, channels=classes)
        self.svls_kernel = self.svls_kernel.cuda()

    def forward(self, inputs, labels):

        # print (inputs.shape, labels.shape)

        with torch.no_grad():
            oh_labels = (labels[...,None] == self.cls_idx).permute(0,3,1,2)
            # print (oh_labels.shape)
            # b, c, h, w = oh_labels.shape
            x = oh_labels.float()
            x = F.pad(x, (1,1,1,1), mode='replicate')
            svls_labels = self.svls_layer(x)/self.svls_kernel.sum()
        return (- svls_labels * F.log_softmax(inputs, dim=1)).sum(dim=1).mean()