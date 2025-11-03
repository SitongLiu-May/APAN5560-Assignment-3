
import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid, save_image
import os

def generate_samples(generator, epoch=None, device="cpu", latent_dim=100, num_samples=16, save_dir="artifacts"):
    """
    Generates and saves sample images from the trained generator.
    """
    generator.eval()
    os.makedirs(save_dir, exist_ok=True)

    z = torch.randn(num_samples, latent_dim, device=device)
    with torch.no_grad():
        fake_imgs = generator(z).detach().cpu()

    grid = make_grid(fake_imgs, nrow=4, normalize=True, value_range=(-1, 1))

    # Optionally save to disk
    if epoch is not None:
        save_path = os.path.join(save_dir, f"samples_epoch_{epoch:02d}.png")
        save_image(grid, save_path)
        print(f"🖼️  Saved sample image grid: {save_path}")

    # Also display inline (optional)
    plt.figure(figsize=(6,6))
    plt.imshow(grid.permute(1, 2, 0))
    plt.axis("off")
    plt.show()
