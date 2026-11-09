
import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mdff_demo import MDFF, demo_image, normalize_map


class HRAR(nn.Module):
    """
    Haze Region-Aware Refinement based on the paper's
    HRE, channel attention and spatial attention equations.

    Uses 1x1 convolutions for the Q/K/V projections.
    """

    def __init__(self, channels=3, reduction=4):
        super().__init__()

        hidden = max(1, channels // reduction)

        # Haze Region Estimator
        self.hre = nn.Sequential(
            nn.Conv2d(channels, 1, 3, padding=1),
            nn.Sigmoid()
        )

        # Global channel descriptor
        self.channel_mlp = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1)
        )

        # Channel-stage Q, K, V
        self.qc = nn.Conv2d(channels, channels, 1)
        self.kc = nn.Conv2d(channels, channels, 1)
        self.vc = nn.Conv2d(1, channels, 1)

        # Spatial attention descriptor
        self.spatial_descriptor = nn.Sequential(
            nn.Conv2d(2, 1, 3, padding=1),
            nn.Sigmoid()
        )

        # Spatial-stage Q, K, V
        self.qs = nn.Conv2d(channels, channels, 1)
        self.ks = nn.Conv2d(1, channels, 1)
        self.vs = nn.Conv2d(1, channels, 1)

    def forward(self, x):
        # Haze estimation
        haze_map = self.hre(x)

        # Stage 1: Channel refinement
        ac = self.channel_mlp(x)

        qc = self.qc(x)
        kc = self.kc(ac)
        vc = self.vc(haze_map)

        alpha_c = torch.sigmoid(qc * kc)
        xc = x + alpha_c * vc

        # Stage 2: Spatial refinement
        max_map = torch.amax(xc, dim=1, keepdim=True)
        mean_map = torch.mean(xc, dim=1, keepdim=True)

        fs = torch.cat([max_map, mean_map], dim=1)
        spatial_map = self.spatial_descriptor(fs)

        qs = self.qs(xc)
        ks = self.ks(spatial_map)
        vs = self.vs(haze_map)

        alpha_s = torch.sigmoid(qs * ks)
        xs = xc + alpha_s * vs

        return haze_map, alpha_c, spatial_map, alpha_s, xs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--size", type=int, default=256)
    args = parser.parse_args()

    torch.manual_seed(42)
    torch.set_num_threads(2)

    if args.image:
        image = Image.open(args.image).convert("RGB")
        image = image.resize((args.size, args.size))
        rgb = np.asarray(image, dtype=np.float32) / 255.0
    else:
        rgb = demo_image(args.size)

    tensor = (
        torch.from_numpy(rgb)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .float()
    )

    mdff = MDFF(channels=3).eval()
    hrar = HRAR(channels=3).eval()

    with torch.no_grad():
        _, _, _, fused = mdff(tensor)

        haze_map, alpha_c, spatial_map, alpha_s, refined = (
            hrar(fused)
        )

    fig, axes = plt.subplots(1, 6, figsize=(20, 4))

    axes[0].imshow(rgb)
    axes[0].set_title("Input image")

    axes[1].imshow(normalize_map(fused), cmap="gray")
    axes[1].set_title("MDFF features")

    axes[2].imshow(
        haze_map[0, 0].cpu().numpy(),
        cmap="inferno", vmin=0, vmax=1
    )
    axes[2].set_title("HRE haze map")

    axes[3].imshow(normalize_map(alpha_c), cmap="gray")
    axes[3].set_title("Channel attention")

    axes[4].imshow(
        spatial_map[0, 0].cpu().numpy(),
        cmap="viridis", vmin=0, vmax=1
    )
    axes[4].set_title("Spatial descriptor")

    axes[5].imshow(normalize_map(refined), cmap="gray")
    axes[5].set_title("HRAR refined features")

    for ax in axes:
        ax.axis("off")

    fig.suptitle(
        "MDFF + Paper-aligned HRAR (untrained)",
        fontsize=14
    )
    fig.tight_layout()

    output = (
        Path(__file__).resolve().parent
        / "hrar_refinement_output.png"
    )
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)

    print("MDFF:", tuple(fused.shape))
    print("HRE:", tuple(haze_map.shape))
    print("Channel attention:", tuple(alpha_c.shape))
    print("Spatial attention:", tuple(alpha_s.shape))
    print("HRAR refined:", tuple(refined.shape))
    print("Saved:", output)
    print("Weights are untrained; maps are not reliable haze estimates.")


if __name__ == "__main__":
    main()
