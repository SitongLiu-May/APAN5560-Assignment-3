# helper/model.py
# ------------------------------------------------------------
# Model definitions for the helper library
# Contains GAN (Generator & Discriminator)
# ------------------------------------------------------------

import torch
import torch.nn as nn


# ----------------------------
# 1️⃣  Generator
# ----------------------------
class Generator(nn.Module):
    """DCGAN-style generator for MNIST (1×28×28 images)."""
    def __init__(self, latent_dim: int = 100, img_shape: tuple = (1, 28, 28)):
        super().__init__()
        self.latent_dim = latent_dim
        self.img_shape = img_shape

        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256 * 7 * 7, bias=False),
            nn.BatchNorm1d(256 * 7 * 7),
            nn.ReLU(True),

            nn.Unflatten(1, (256, 7, 7)),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),  # 7→14
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),   # 14→28
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.Conv2d(64, img_shape[0], 3, 1, 1, bias=False),
            nn.Tanh()                                           # output [-1,1]
        )

    def forward(self, z):
        return self.model(z)


# ----------------------------
# 2️⃣  Discriminator
# ----------------------------
class Discriminator(nn.Module):
    """DCGAN-style discriminator for MNIST (1×28×28 images)."""
    def __init__(self, img_shape: tuple = (1, 28, 28)):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(img_shape[0], 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        return self.model(img)


# ----------------------------
# 3️⃣  Weight Initialization
# ----------------------------
def weights_init(m):
    """Normal initialization as used in DCGAN paper."""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if getattr(m, "bias", None) is not None:
            nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.zeros_(m.bias)


# ----------------------------
# 4️⃣  Optional model selector
# ----------------------------
def get_model(model_name: str):
    """Factory function so other modules can easily request models."""
    if model_name.upper() == "GAN":
        return Generator(), Discriminator()
    else:
        raise ValueError(f"Unknown model name: {model_name}")
