from helper.model import Generator, Discriminator, weights_init
from helper.trainer import train_gan
from helper.generator import generate_samples
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize and train GAN
train_gan(epochs=5, device=device)

# Generate final sample images
gen = Generator().to(device)
gen.load_state_dict(torch.load("gan_model.pth", map_location=device))
generate_samples(gen, epoch=5, device=device)
