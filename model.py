
import argparse
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mdff_demo import MDFF
from hrar import HRAR
from hgfm import HGFM


# =====================================
# 1. AERIAL HAZE REMOVAL BLOCK
# =====================================

class AHRBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.mdff = MDFF(channels=channels)
        self.hrar = HRAR(channels=channels)
        self.hgfm = HGFM(
            channels=channels,
            expansion=2
        )

    def forward(self, x):
        _, _, _, fused = self.mdff(x)
        _, _, _, _, refined = self.hrar(fused)

        y = x + refined

        _, _, _, gated = self.hgfm(y)
        z = y + gated

        return z


# =====================================
# 2. COMPLETE DEHAZING NETWORK
# =====================================

class DehazingNet(nn.Module):
    def __init__(self, base_channels=16):
        super().__init__()

        c = base_channels

        self.stem = nn.Conv2d(3, c, 3, padding=1)
        self.encoder = AHRBlock(c)

        self.downsample = nn.Conv2d(
            c, 2*c, 3, stride=2, padding=1
        )

        self.bottleneck = AHRBlock(2*c)

        self.up_project = nn.Conv2d(
            2*c, c, 3, padding=1
        )

        self.skip_fusion = nn.Conv2d(
            2*c, c, 1
        )

        self.decoder = AHRBlock(c)

        self.rgb_head = nn.Conv2d(
            c, 3, 3, padding=1
        )

        nn.init.zeros_(self.rgb_head.weight)
        nn.init.zeros_(self.rgb_head.bias)

    def forward(self, image):
        features = self.stem(image)
        encoded = self.encoder(features)

        down = self.downsample(encoded)
        bottleneck = self.bottleneck(down)

        up = F.interpolate(
            bottleneck,
            size=encoded.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        up = self.up_project(up)

        combined = torch.cat([up, encoded], dim=1)
        combined = self.skip_fusion(combined)

        decoded = self.decoder(combined)
        correction = self.rgb_head(decoded)

        return image + correction


# =====================================
# 3. SELECT IMAGE FROM PC
# =====================================

def select_image():

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    image_path = filedialog.askopenfilename(
        title="Select an image to dehaze",
        filetypes=[
            ("Image files", "*.png *.jpg *.jpeg *.bmp *.webp"),
            ("All files", "*.*")
        ]
    )

    root.destroy()

    return image_path


# =====================================
# 4. LOAD IMAGE
# =====================================

def load_image(path, size):

    image = Image.open(path).convert("RGB")

    original_size = image.size

    image = image.resize(
        (size, size),
        Image.Resampling.LANCZOS
    )

    rgb = np.asarray(
        image, dtype=np.float32
    ) / 255.0

    tensor = torch.from_numpy(rgb)
    tensor = tensor.permute(2, 0, 1)
    tensor = tensor.unsqueeze(0).float()

    return rgb, tensor, original_size


# =====================================
# 5. DEHAZE IMAGE
# =====================================

def dehaze(image_path, weights_path, size):

    print("\nLoading image...")

    rgb, tensor, original_size = load_image(
        image_path, size
    )

    model = DehazingNet(base_channels=16)

    if not weights_path.exists():
        raise FileNotFoundError(
            f"Trained model not found:\n{weights_path}"
        )

    state = torch.load(
        weights_path,
        map_location="cpu",
        weights_only=True
    )

    model.load_state_dict(state)
    model.eval()

    print("Trained model loaded successfully.")
    print("Processing image...")

    torch.set_num_threads(2)

    with torch.no_grad():
        output = model(tensor)

    output_rgb = (
        output.clamp(0, 1)
        .squeeze(0)
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )

    # Resize restored image to original dimensions
    restored = Image.fromarray(
        (output_rgb * 255).astype(np.uint8)
    )

    restored = restored.resize(
        original_size,
        Image.Resampling.LANCZOS
    )

    output_dir = Path(__file__).resolve().parent / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    restored_path = output_dir / "restored_rgb.png"
    comparison_path = output_dir / "comparison.png"

    restored.save(restored_path)

    # Original image at its original resolution
    original = Image.open(image_path).convert("RGB")

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].imshow(original)
    axes[0].set_title("Original Image")

    axes[1].imshow(restored)
    axes[1].set_title("Dehazed Image")

    for ax in axes:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(
        comparison_path,
        dpi=160,
        bbox_inches="tight"
    )
    plt.close(fig)

    print("\n========== RESULTS ==========")
    print("Input image:", image_path)
    print("Input size:", original_size)
    print("Model input size:", size)
    print("Restored image:", restored_path)
    print("Comparison:", comparison_path)
    print("=============================\n")

    return restored_path


# =====================================
# 6. MAIN APPLICATION
# =====================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--weights", type=str, default=None)

    args = parser.parse_args()

    project_dir = Path(__file__).resolve().parent

    if args.weights:
        weights_path = Path(args.weights)
    else:
        weights_path = project_dir / "outputs" / "best_model.pth"

    # Open file picker if no image path is supplied
    if args.image:
        image_path = args.image
    else:
        print("Choose an image from your computer...")
        image_path = select_image()

    if not image_path:
        print("No image selected. Exiting.")
        return

    try:
        restored_path = dehaze(
            image_path,
            weights_path,
            args.size
        )

        root = tk.Tk()
        root.withdraw()
        messagebox.showinfo(
            "Dehazing Complete",
            f"Your image has been dehazed successfully!\n\n"
            f"Saved at:\n{restored_path}"
        )
        root.destroy()

    except Exception as e:
        print("Error:", e)

        root = tk.Tk()
        root.withdraw()
        messagebox.showerror(
            "Dehazing Error",
            str(e)
        )
        root.destroy()


if __name__ == "__main__":
    main()
