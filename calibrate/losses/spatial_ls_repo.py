import torch
import torch.nn.functional as F
import math

def get_gaussian_kernel_2d(ksize=0, sigma=0):
    x_grid = torch.arange(ksize).repeat(ksize).view(ksize, ksize)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()
    mean = (ksize - 1)/2.
    variance = sigma**2.
    gaussian_kernel = (1./(2.*math.pi*variance + 1e-16)) * torch.exp( 
        -torch.sum((xy_grid - mean)**2., dim=-1) / (2*variance + 1e-16)
        )
    return gaussian_kernel / torch.sum(gaussian_kernel)

class get_svls_filter_2d(torch.nn.Module):
    def __init__(self, ksize=3, sigma=0, channels=0):
        super(get_svls_filter_2d, self).__init__()
        gkernel = get_gaussian_kernel_2d(ksize=ksize, sigma=sigma)
        neighbors_sum = (1 - gkernel[1,1]) + 1e-16
        gkernel[int(ksize/2), int(ksize/2)] = neighbors_sum
        self.svls_kernel = gkernel / neighbors_sum
        svls_kernel_2d = self.svls_kernel.view(1, 1, ksize, ksize)
        svls_kernel_2d = svls_kernel_2d.repeat(channels, 1, 1, 1)
        padding = int(ksize/2)
        self.svls_layer = torch.nn.Conv2d(in_channels=channels, out_channels=channels,
                                    kernel_size=ksize, groups=channels,
                                    bias=False, padding=padding, padding_mode='replicate')
        self.svls_layer.weight.data = svls_kernel_2d
        self.svls_layer.weight.requires_grad = False
    def forward(self, x):
        return self.svls_layer(x) / self.svls_kernel.sum()

class CELossWithSVLS_2D(torch.nn.Module):
    def __init__(self, classes=None, sigma=1, ksize=3):
        super(CELossWithSVLS_2D, self).__init__()
        self.cls = torch.tensor(classes)
        self.svls_layer = get_svls_filter_2d(ksize=3, sigma=sigma, channels=self.cls)

    def forward(self, inputs, labels):
        oh_labels = F.one_hot(labels.to(torch.int64), num_classes = self.cls).contiguous().permute(0,3,1,2).float()
        svls_labels = self.svls_layer(oh_labels)
        return (- svls_labels * F.log_softmax(inputs, dim=1)).sum(dim=1).mean()