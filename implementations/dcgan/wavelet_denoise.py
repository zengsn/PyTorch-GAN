"""
https://notebook.community/t-vi/pytorch-tvmisc/misc/2D-Wavelet-Transform
"""

import io
import torch
import pywt
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
from PIL import Image
import urllib.request

URL = 'https://upload.wikimedia.org/wikipedia/commons/thumb/b/bc/Zuse-Z4-Totale_deutsches-museum.jpg/315px-Zuse-Z4-Totale_deutsches-museum.jpg'

# %matplotlib inline
print(pywt.families())
w_families = pywt.families()
w_name_list = []
for w_fname in w_families:
    w_list = pywt.wavelist(w_fname)
    w_name_list += w_list
print(w_name_list)
print(w_name_list.index('bior2.2'))

w = pywt.Wavelet('bior2.2')
plt.plot(w.dec_hi[::-1], label="dec hi")
plt.plot(w.dec_lo[::-1], label="dec lo")
plt.plot(w.rec_hi, label="rec hi")
plt.plot(w.rec_lo, label="rec lo")
plt.title("Bior 2.2 Wavelets")
plt.legend()

dec_hi = torch.tensor(w.dec_hi[::-1])
dec_lo = torch.tensor(w.dec_lo[::-1])
rec_hi = torch.tensor(w.rec_hi)
rec_lo = torch.tensor(w.rec_lo)


filters = torch.stack([
    dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1),
    dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1),
    dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1),
    dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)], dim=0)


inv_filters = torch.stack([
    rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1),
    rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1),
    rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1),
    rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1)], dim=0)


def wt(vimg, levels=1):
    h = vimg.size(2)
    w = vimg.size(3)
    padded = torch.nn.functional.pad(vimg, (2, 2, 2, 2))
    res = torch.nn.functional.conv2d(padded, filters[:, None], stride=2)
    if levels > 1:
        res[:, :1] = wt(res[:, :1], levels - 1)
    res = res.view(-1, 2, h // 2, w // 2).transpose(1, 2).contiguous().view(-1, 1, h, w)
    return res


def iwt(vres, levels=1):
    h = vres.size(2)
    w = vres.size(3)
    res = vres.view(-1, h // 2, 2, w // 2).transpose(1, 2).contiguous().view(-1, 4, h // 2, w // 2).clone()
    if levels > 1:
        res[:, :1] = iwt(res[:, :1], levels=levels - 1)
    res = torch.nn.functional.conv_transpose2d(res, inv_filters[:, None], stride=2)
    res = res[:, :, 2:-2, 2:-2]
    return res


imgraw = Image.open(io.BytesIO(urllib.request.urlopen(URL).read())).resize((256, 256))
img = np.array(imgraw).mean(2) / 255
img = torch.from_numpy(img).float()
plt.figure()
plt.imshow(img, cmap=plt.cm.gray)

vimg = img[None, None]
res = wt(vimg, 4)
plt.figure()
plt.imshow(res[0, 0].data.numpy(), cmap=plt.cm.gray)
rec = iwt(res, levels=4)
plt.imshow(rec[0, 0].data.numpy(), cmap=plt.cm.gray)
