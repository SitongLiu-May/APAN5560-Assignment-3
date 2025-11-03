import torch
from tqdm import tqdm

def train_gan(
    generator,
    discriminator,
    data_loader,
    optimizer_G,
    optimizer_D,
    criterion,
    device="cpu",
    epochs=10,
    latent_dim=100,
    save_interval=1,
    sample_fn=None
):
    """
    Trains a Generative Adversarial Network (GAN).

    Parameters
    ----------
    generator : nn.Module
        The Generator model (from helper.model)
    discriminator : nn.Module
        The Discriminator model (from helper.model)
    data_loader : DataLoader
        Training dataset loader (e.g. MNIST)
    optimizer_G : torch.optim.Optimizer
        Optimizer for the generator
    optimizer_D : torch.optim.Optimizer
        Optimizer for the discriminator
    criterion : torch.nn.modules.loss
        Loss function (usually BCELoss)
    device : str
        'cuda', 'mps', or 'cpu'
    epochs : int
        Number of training epochs
    latent_dim : int
        Dimension of latent vector z
    save_interval : int
        How often to trigger sample saving (epochs)
    sample_fn : callable
        Optional callback for saving sample images each epoch
    """

    print(f"🚀 Starting GAN training on {str(device).upper()} for {epochs} epochs...")

    generator.train()
    discriminator.train()

    for epoch in range(1, epochs + 1):
        g_loss_total, d_loss_total = 0.0, 0.0

        for imgs, _ in tqdm(data_loader, desc=f"Epoch {epoch}/{epochs}"):
            imgs = imgs.to(device)
            batch_size = imgs.size(0)

            # ----------------------
            # 1️⃣ Train Discriminator
            # ----------------------
            optimizer_D.zero_grad()

            z = torch.randn(batch_size, latent_dim, device=device)
            fake_imgs = generator(z).detach()  # detach so G not updated here

            real_validity = discriminator(imgs)
            fake_validity = discriminator(fake_imgs)

            real_loss = criterion(real_validity, torch.ones_like(real_validity))
            fake_loss = criterion(fake_validity, torch.zeros_like(fake_validity))
            d_loss = (real_loss + fake_loss) / 2

            d_loss.backward()
            optimizer_D.step()

            # ----------------------
            # 2️⃣ Train Generator
            # ----------------------
            optimizer_G.zero_grad()

            z = torch.randn(batch_size, latent_dim, device=device)
            gen_imgs = generator(z)
            validity = discriminator(gen_imgs)
            g_loss = criterion(validity, torch.ones_like(validity))

            g_loss.backward()
            optimizer_G.step()

            g_loss_total += g_loss.item()
            d_loss_total += d_loss.item()

        # ----------------------
        # 🔄 Epoch summary
        # ----------------------
        avg_g = g_loss_total / len(data_loader)
        avg_d = d_loss_total / len(data_loader)
        print(f"📈 Epoch [{epoch}/{epochs}] | D_loss: {avg_d:.4f} | G_loss: {avg_g:.4f}")

        # Optional sample callback (e.g. save generated images)
        if sample_fn and (epoch % save_interval == 0):
            try:
                sample_fn(generator, epoch, device)
            except Exception as e:
                print(f"⚠️  Warning: Could not save sample for epoch {epoch}: {e}")

    print("✅ GAN training complete.")
    return generator, discriminator
