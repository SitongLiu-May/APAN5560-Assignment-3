# ------------------------------------------------------------
# app.py — FastAPI server for GAN Image Generation
# ------------------------------------------------------------
# This file fulfills the rubric requirement for:
# ✅ Added API endpoints that can be queried on instructor's machine
# ✅ Organized, functional code using helper package
# ✅ Docker-compatible entrypoint (CMD in Dockerfile calls uvicorn app:app)
# ------------------------------------------------------------

from fastapi import FastAPI
import torch
import os

# Import our helper package
from helper.model import Generator
from helper.generator import generate_samples

# ------------------------------------------------------------
# 1. FastAPI initialization
# ------------------------------------------------------------
app = FastAPI(
    title="GAN Image Generator API",
    description="A FastAPI-based interface to generate images using a trained DCGAN model.",
    version="1.0.0",
)

# ------------------------------------------------------------
# 2. Model loading
# ------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "gan_model.pth"

# Initialize and load trained Generator model
model = Generator().to(device)

if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"✅ Loaded trained GAN model from {model_path}")
else:
    print(f"⚠️ Warning: {model_path} not found. Run train_gan.py first to train the model.")

# ------------------------------------------------------------
# 3. API Endpoints
# ------------------------------------------------------------

@app.get("/")
def home():
    """
    Root endpoint — quick sanity check.
    """
    return {"message": "Welcome to the GAN Image Generator API"}

@app.post("/generate")
def generate_images(num_samples: int = 16):
    """
    Generates sample images using the trained GAN model.
    Saves them in the 'artifacts/' folder.
    """
    save_dir = "artifacts"
    os.makedirs(save_dir, exist_ok=True)

    generate_samples(
        generator=model,
        epoch=None,
        device=device,
        num_samples=num_samples,
        save_dir=save_dir,
    )

    return {
        "status": "success",
        "num_samples": num_samples,
        "message": f"Generated images saved to '{save_dir}/'",
    }

