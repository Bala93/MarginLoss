import torch
import torch.nn.functional as F
import math

def get_svls_filter_2d(kernel_size=3, sigma=1, channels=4):
    # Create a x, y, z coordinate grid of shape (kernel_size, kernel_size, kernel_size, 3)
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    # x_grid = x_coord.repeat(kernel_size*kernel_size).view(kernel_size, kernel_size, kernel_size)
    # print (x_grid.shape)
    y_grid = x_grid.t()
    # print (y_grid.shape)
    # y_grid  = y_grid_2d.repeat(kernel_size,1).view(kernel_size, kernel_size)
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()
    # print (xy_grid.shape)
    mean = (kernel_size - 1)/2.
    variance = sigma**2.
    # Calculate the 3-dimensional gaussian kernel
    gaussian_kernel = (1./(2.*math.pi*variance + 1e-16)) * torch.exp(
                            -torch.sum((xy_grid - mean)**2., dim=-1) / (2*variance + 1e-16))
    
    # print (gaussian_kernel.shape)

    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
    neighbors_sum = 1 - gaussian_kernel[1,1]
    gaussian_kernel[1,1] = neighbors_sum
    svls_kernel_2d = gaussian_kernel / neighbors_sum
    
    # print (svls_kernel_2d.shape)

    # Reshape to 3d depthwise convolutional weight
    svls_kernel_2d = svls_kernel_2d.view(1, 1, kernel_size, kernel_size)
    svls_kernel_2d = svls_kernel_2d.repeat(channels, 1, 1, 1)
    
    # print (svls_kernel_2d.shape)
    
    svls_filter_2d = torch.nn.Conv2d(in_channels=channels, out_channels=channels,
                                kernel_size=kernel_size, groups=channels,
                                bias=False, padding=0)
    
    
    svls_filter_2d.weight.data = svls_kernel_2d
    svls_filter_2d.weight.requires_grad = False 
    
    return svls_filter_2d, svls_kernel_2d[0]


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