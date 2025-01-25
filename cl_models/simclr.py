import torch.nn as nn
from PIL import Image

import skimage as sk
import numpy as np
from torchvision import transforms

class Brightness:
    def __init__(self, severity=5):
        self.severity = severity
        self.c = [.05, .1, .15, .2, .3][severity - 1]

    def __call__(self, x):
        #print('x.shape', x.shape)
        if x.shape[2] != 3:
            x= x.permute(1, 2, 0)
        x = np.array(x) / 255.
        x = sk.color.rgb2hsv(x)
        x[:, :, 2] = np.clip(x[:, :, 2] + self.c, 0, 1)
        x = sk.color.hsv2rgb(x)
        return Image.fromarray((np.clip(x, 0, 1) * 255).astype(np.uint8))

class StrongCrop:
    def __init__(self, img_size=32, severity=5):
        crop_size = [224, 192, 160, 128, 96][severity - 1]
        ratio = crop_size / 224 
        self.cifar_crops = int(img_size * ratio) 
        self.img_size = img_size

    def __call__(self, x):
        crop_size = self.cifar_crops
        transform = transforms.Compose([
            transforms.RandomCrop(crop_size),
            transforms.Resize((self.img_size, self.img_size)),
        ])
        return transform(x) 


class SimCLR(nn.Module):
    def __init__(self, base_encoder, projection_dim, modified_conv1=False):
        super().__init__()
        self.enc = base_encoder#(pre_trained=False)
        self.feature_dim = self.enc.fc.in_features
        if modified_conv1:
            self.enc.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
            self.enc.maxpool = nn.Identity()
        # Replace the fc layer with an Identity function        
        self.enc.fc = nn.Identity()
        self.projection_dim = projection_dim
        # We use a MLP with one hidden layer to obtain z_i = g(h_i) = W(2)σ(W(1)h_i) where σ is a ReLU non-linearity.
        self.projector = nn.Sequential(
            nn.Linear(self.feature_dim, 2048, bias=False),
            nn.ReLU(),
            nn.Linear(2048, projection_dim, bias=False),
        )

    def forward(self, x):
        h = self.enc(x)
        z = self.projector(h)
        return h, z

