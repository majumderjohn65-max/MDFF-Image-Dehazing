
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
from hrar import HRAR


class HGFM(nn.Module):
    """
    Hadamard-Gated Feature Modulation.

    Based on Section II-C of the IEEE paper.
    Expansion ratio = 2 for this demonstration.
    """

    def __init__(self, channels=3, expansion=2):
        super().__init__()

        expanded = channels * expansion

        if expanded % 2 != 0:
            raise ValueError(
                "Expanded channel count must be even."
            )

        self.norm = nn.LayerNorm(channels)

        # Expand channel dimension
        self.expand = nn.Conv2d(
            channels, expanded, kernel_size=1
        )

        # Project gated features to original channels
        self.project = nn.Conv2d(
            expanded // 2, channels, kernel_size=1
        )

    def forward(self, x):
        # Layer normalization across channels
        x_norm = self.norm(
            x.permute(0, 2, 3, 1)
        ).permute(0, 3, 1, 2)

        # Channel expansion
        expanded = self.expand(x_norm)

        # Split expanded features into two equal groups
        f1, f2 = torch.chunk(expanded, 2, dim=1)

        # Hadamard (elementwise) multiplication
        gated = f1 * f2

        # Return to original channel dimension
        output = self.project(gated)

        return f1, f2, gated, output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--size", type=int, default=256)
    args = parser.parse_args()

    torch.manual_seed(42)
    torch.set_num_threads(2)

    # Load input image
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

    # Initialize the three modules
    mdff = MDFF(channels=3).eval()
    hrar = HRAR(channels=3).eval()
    hgfm = HGFM(channels=3, expansion=2).eval()

    with torch.no_grad():
        # Stage 1: MDFF
        _, _, _, fused = mdff(tensor)

        # Stage 2: HRAR
        _, _, _, _, refined = hrar(fused)

        # Paper's AHR residual connections
        y = refined + tensor

        # Stage 3: HGFM
        f1, f2, gated, hgfm_output = hgfm(y)

        # Final AHR block output
        z = y + hgfm_output

    # Visualize all stages
    fig, axes = plt.subplots(1, 7, figsize=(23, 4))

    panels = [
        ("Input image", rgb),
        ("MDFF", normalize_map(fused)),
        ("HRAR", normalize_map(refined)),
        ("HGFM F1", normalize_map(f1)),
        ("HGFM F2", normalize_map(f2)),
        ("Hadamard gated", normalize_map(gated)),
        ("AHR output", normalize_map(z)),
    ]

    for i, (title, data) in enumerate(panels):
        if i == 0:
            axes[i].imshow(data)
        else:
            axes[i].imshow(
                data, cmap="gray", vmin=0, vmax=1
            )

        axes[i].set_title(title)
        axes[i].axis("off")

    fig.suptitle(
        "MDFF + HRAR + HGFM (untrained AHR block)",
        fontsize=14
    )
    fig.tight_layout()

    output_path = (
        Path(__file__).resolve().parent
        / "hgfm_output.png"
    )

    fig.savefig(
        output_path, dpi=160, bbox_inches="tight"
    )
    plt.close(fig)

    print("Input:", tuple(tensor.shape))
    print("MDFF:", tuple(fused.shape))
    print("HRAR:", tuple(refined.shape))
    print("F1:", tuple(f1.shape))
    print("F2:", tuple(f2.shape))
    print("Hadamard gated:", tuple(gated.shape))
    print("HGFM output:", tuple(hgfm_output.shape))
    print("AHR final output:", tuple(z.shape))
    print("Saved:", output_path)
    print(
        "Note: The modules are untrained. "
        "The AHR output is not a dehazed RGB image."
    )


if __name__ == "__main__":
    main()
