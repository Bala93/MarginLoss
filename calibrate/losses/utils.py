import torch
import torch.nn.functional as F
import math

'''
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
    
'''


def get_gaussian_kernel_2d(ksize=3, sigma=1):
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
    def __init__(self, ksize=3, sigma=1, channels=0):
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
