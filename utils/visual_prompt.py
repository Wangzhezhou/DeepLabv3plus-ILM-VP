import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ExpansiveVisualPrompt(nn.Module):
    def __init__(self, out_size, mask, init='zero', normalize=None):
        super(ExpansiveVisualPrompt, self).__init__()
        assert mask.shape[0] == mask.shape[1]
        in_size = mask.shape[0]
        self.out_size = out_size
        if init == "zero":
            self.program = torch.nn.Parameter(data=torch.zeros(3, out_size, out_size))
        elif init == "randn":
            self.program = torch.nn.Parameter(data=torch.randn(3, out_size, out_size))
        else:
            raise ValueError("init method not supported")
        self.normalize = normalize

        l_pad = (self.out_size - in_size) // 2
        r_pad = self.out_size - in_size - l_pad
        t_pad = l_pad
        b_pad = r_pad

        mask = np.repeat(np.expand_dims(mask, 0), repeats=3, axis=0)
        mask = torch.Tensor(mask)
        self.register_buffer("mask", F.pad(mask, (l_pad, r_pad, t_pad, b_pad), value=1))

    def forward(self, x):
        # Ensure x is a 4-dimensional tensor
        # Ensure x is a 4-dimensional tensor
        while len(x.shape) < 4:
            x = x.unsqueeze(0)
        while len(x.shape) > 4:
            x = x.squeeze(1)  # Remove the second dimension, which seems to be the extra one


        if len(x.shape) != 4:
            print("Unexpected tensor shape:", x.shape)
            return x  # or raise an error

        batch_size, c, h, w = x.shape

        l_pad = (self.out_size - w) // 2
        r_pad = self.out_size - w - l_pad
        t_pad = (self.out_size - h) // 2
        b_pad = self.out_size - h - t_pad

        # Pad x and add it to the program-mask product
        x = F.pad(x, (l_pad, r_pad, t_pad, b_pad), value=0)

        for i in range(batch_size):
            for j in range(c):
                x[i, j, :, :] += torch.sigmoid(self.program[j]) * self.mask[j]

        if self.normalize is not None:
            x = self.normalize(x)
        return x

