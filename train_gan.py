import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from helper_lib.model import Generator, Discriminator, weights_init
from helper_lib.trainer import train_gan
from helper_lib.generator import generate_samples

# ==================================================
# Configuration
# ==================================================
batch_size = 128
epochs = 5
lr = 0.0002
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("✅ Using Apple Metal (MPS) accelerator.")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("✅ Using NVIDIA CUDA GPU.")
else:
    device = torch.device("cpu")
    print("⚙️ Using CPU (no GPU available).")


# ==================================================
# Data Loader (MNIST)
# ==================================================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# ==================================================
# Initialize Models
# ==================================================
latent_dim = 100
generator = Generator(latent_dim).to(device)
discriminator = Discriminator().to(device)

generator.apply(weights_init)
discriminator.apply(weights_init)

# ==================================================
# Optimizers and Loss
# ==================================================
optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
criterion = nn.BCELoss()

# ==================================================
# Train
# ==================================================
train_gan(
    generator,
    discriminator,
    data_loader,
    optimizer_G,
    optimizer_D,
    criterion,
    device=device,
    epochs=epochs
)

# ==================================================
# Generate final samples
# ==================================================
generate_samples(generator, epoch=epochs, device=device)

# Save model
torch.save(generator.state_dict(), "gan_model.pth")
print("✅ Training complete. Model saved to generator.pth")
