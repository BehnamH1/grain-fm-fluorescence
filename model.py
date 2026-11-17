"""U-Net model (encoder-decoder with skip connections, after the pix2pix generator
of Isola et al. 2017). 16 convolution layers, 54,413,955 parameters.
Input and output are 3 x 256 x 256 images scaled to [-1, 1].
Also the shared steps: image to tensor, running the model on one image, FM ratio."""
import cv2
import numpy as np
import torch
import torch.nn as nn

IMG_SIZE = 256


def to_tensor(img, interp=cv2.INTER_LINEAR):
    """BGR image -> 3 x 256 x 256 tensor in [-1, 1]. Labels use INTER_NEAREST."""
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=interp)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = (img.astype(np.float32) / 127.5) - 1.0
    return torch.from_numpy(img.transpose(2, 0, 1))


def predict_image(model, img):
    """Run the model on one BGR image and return the output at the input size."""
    h, w = img.shape[:2]
    with torch.no_grad():
        y = model(to_tensor(img).unsqueeze(0))
    y = ((y[0].numpy() + 1.0) * 127.5).clip(0, 255).astype(np.uint8).transpose(1, 2, 0)
    return cv2.resize(cv2.cvtColor(y, cv2.COLOR_RGB2BGR), (w, h), interpolation=cv2.INTER_NEAREST)


def fm_ratio(bgr):
    """FM pixels (green > 150, red and blue < 100) over kernel pixels (gray level > 10)."""
    kernel = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) > 10
    if kernel.sum() == 0:
        return 0.0
    r, g, b = bgr[:, :, 2], bgr[:, :, 1], bgr[:, :, 0]
    return float(((g > 150) & (r < 100) & (b < 100)).sum() / kernel.sum())


def load_model(path):
    model = UNet()
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model


def down_block(in_ch, out_ch, normalize=True, dropout=0.0):
    layers = [nn.Conv2d(in_ch, out_ch, 4, stride=2, padding=1, bias=False)]
    if normalize:
        layers.append(nn.BatchNorm2d(out_ch))
    layers.append(nn.LeakyReLU(0.2, inplace=True))
    if dropout:
        layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


def up_block(in_ch, out_ch, dropout=0.0):
    layers = [nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1, bias=False),
              nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)]
    if dropout:
        layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.d1 = down_block(3, 64, normalize=False)
        self.d2 = down_block(64, 128)
        self.d3 = down_block(128, 256)
        self.d4 = down_block(256, 512, dropout=0.5)
        self.d5 = down_block(512, 512, dropout=0.5)
        self.d6 = down_block(512, 512, dropout=0.5)
        self.d7 = down_block(512, 512, dropout=0.5)
        self.d8 = down_block(512, 512, normalize=False)
        self.u1 = up_block(512, 512, dropout=0.5)
        self.u2 = up_block(512 + 512, 512, dropout=0.5)
        self.u3 = up_block(512 + 512, 512, dropout=0.5)
        self.u4 = up_block(512 + 512, 512)
        self.u5 = up_block(512 + 512, 256)
        self.u6 = up_block(256 + 256, 128)
        self.u7 = up_block(128 + 128, 64)
        self.final = nn.Sequential(
            nn.ConvTranspose2d(64 + 64, 3, 4, stride=2, padding=1),
            nn.Tanh())

    def forward(self, x):
        d1 = self.d1(x); d2 = self.d2(d1); d3 = self.d3(d2); d4 = self.d4(d3)
        d5 = self.d5(d4); d6 = self.d6(d5); d7 = self.d7(d6); d8 = self.d8(d7)
        u1 = self.u1(d8)
        u2 = self.u2(torch.cat([u1, d7], 1)); u3 = self.u3(torch.cat([u2, d6], 1))
        u4 = self.u4(torch.cat([u3, d5], 1)); u5 = self.u5(torch.cat([u4, d4], 1))
        u6 = self.u6(torch.cat([u5, d3], 1)); u7 = self.u7(torch.cat([u6, d2], 1))
        return self.final(torch.cat([u7, d1], 1))
