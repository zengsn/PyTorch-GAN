# Inspired by 1D wavelet transformation
# at https://github.com/v0lta/Wavelet-network-compression/blob/master/wavelet_learning/wavelet_linear.py
import torch
import numpy as np
from torch.nn.parameter import Parameter
import pywt  # based on PyWavelets


class WaveletLayer2(torch.nn.Module):
    """
    Create a learn-able 2D Wavelet layer as described here:
    https://arxiv.org/pdf/2004.09569.pdf
    The weights are parametrized by S*W*G*P*W*B
    With S,G,B diagonal matrices, P a random permutation and W a
    learnable-wavelet transform.
    """

    def __init__(self, wavelet_name='bior2.2', scales=5, p_drop=0.5):
        super().__init__()
        print("Wavelet 2D transformation using:", wavelet_name)
        self.wavelet_name = wavelet_name
        self.wavelet = pywt.Wavelet(wavelet_name)

    def filters(self):
        w = self.wavelet
        dec_hi = torch.tensor(w.dec_hi[::-1])
        dec_lo = torch.tensor(w.dec_lo[::-1])
        return torch.stack([
            dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1),
            dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1),
            dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1),
            dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)], dim=0)

    def wt(self, x, levels=1):
        """ Wavelet Transformation """
        h = x.size(2)
        w = x.size(3)
        padded = torch.nn.functional.pad(x, (2, 2, 2, 2))
        filters = self.filters()
        res = torch.nn.functional.conv2d(padded, filters[:, None], stride=2)
        if levels > 1:
            res[:, :1] = self.wt(res[:, :1], levels - 1)
        res = res.view(-1, 2, h // 2, w // 2).transpose(1, 2).contiguous().view(-1, 1, h, w)
        return res

    def inv_filters(self):
        w = self.wavelet
        rec_hi = torch.tensor(w.rec_hi)
        rec_lo = torch.tensor(w.rec_lo)
        return torch.stack([
            rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1),
            rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1),
            rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1),
            rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1)], dim=0)

    def iwt(self, x, levels=1):
        """ Inverse Wavelet Transformation """
        h = x.size(2)
        w = x.size(3)
        res = x.view(-1, h // 2, 2, w // 2).transpose(1, 2).contiguous().view(-1, 4, h // 2, w // 2).clone()
        if levels > 1:
            res[:, :1] = self.iwt(res[:, :1], levels=levels - 1)
        inv_filters = self.inv_filters()
        res = torch.nn.functional.conv_transpose2d(res, inv_filters[:, None], stride=2)
        res = res[:, :, 2:-2, 2:-2]
        return res

    def forward(self, x):
        x = self.wt(x)
        x = self.iwt(x)
        return x

    def wavelet_loss(self):
        """ 获取小波变换，用于叠加权重。"""
        pass


