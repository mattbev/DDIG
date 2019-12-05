import torch
import numpy as np


class HPool(torch.nn.Module):
    def __init__(self, channels, height, width, num_bins):
        super(HPool, self).__init__()

        #         self.num_bins = height*width // 100
        self.num_bins = num_bins

        self.channels = channels
        self.height = height
        self.width = width
        #         self.coeff = torch.rand(self.channels, self.height*self.width, requires_grad=True)

        self.coeff = torch.rand(self.channels, self.num_bins, requires_grad=True)

    #         self.coeff = torch.Tensor(self.channels, self.height*self.width)
    #         self.coeff.fill_(1/(self.height*self.width))
    #         self.coeff.requires_grad_(True)

    @staticmethod
    def histogram3D(x, bins):
        N, C, H = x.shape
        x_max = torch.max(x).item()
        x_min = torch.min(x).item()
        tau = np.linspace(x_min, x_max, bins + 1)
        histogram = torch.zeros(N, C, bins)
        x_tanh = torch.tanh(x)

        for b in range(bins):
            if b < bins - 1:
                mask = (tau[b] <= x) & (x < tau[b + 1])
            else:
                mask = (tau[b] <= x)
            histogram[:, :, b] = torch.sum(torch.zeros_like(x).masked_scatter_(mask, x_tanh), dim=2)

        return histogram

    def forward(self, x):
        # Dimension changes from (N, C, H, W) to (N, C, H*W)
        x = x.view(-1, self.channels, self.height * self.width)

        # Sort the vector
        #         x, indexes = torch.sort(x, dim=2)

        # Create a histogram
        y = HPool.histogram3D(x, bins=self.num_bins)

        # Elementwise multiplication with trainable coefficient vector
        y = torch.mul(y, self.coeff)

        # Dimension changes from (N, C, H*W) to (N, C)
        z = torch.sum(y, dim=2)

        return z
