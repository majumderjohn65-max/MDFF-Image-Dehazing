import argparse
from pathlib import Path
import numpy as np
import torch
from torch import nn
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class MDFF(nn.Module):
    def __init__(self, channels=3):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.horizontal = nn.Conv2d(channels, channels, (1, 5), padding=(0, 2), groups=channels, bias=False)
        self.vertical = nn.Conv2d(channels, channels, (5, 1), padding=(2, 0), groups=channels, bias=False)
        self.dilated = nn.Conv2d(channels, channels, 3, padding=3, dilation=3, groups=channels, bias=False)
        self.fusion = nn.Conv2d(3 * channels, channels, 1, bias=False)

    def forward(self, x):
        x = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        h = self.horizontal(x)
        v = self.vertical(x)
        d = self.dilated(x)
        fused = self.fusion(torch.cat([h, v, d], dim=1))
        return h, v, d, fused


def demo_image(size=256):
    y, x = np.mgrid[0:size, 0:size]
    base = np.zeros((size, size, 3), dtype=np.float32)
    base[..., 0] = 0.2 + 0.5 * (x / size)
    base[..., 1] = 0.25 + 0.4 * (y / size)
    base[..., 2] = 0.3
    base[size//3:size//3+12, :] = [0.8, 0.8, 0.75]
    base[:, size//2:size//2+10] = [0.75, 0.75, 0.72]
    for k in range(4):
        y0 = 25 + k * 48
        x0 = 25 + (k % 2) * 65
        base[y0:y0+25, x0:x0+35] = [0.7, 0.35, 0.2]
    haze = 0.12 + 0.45 * (y / size) * (x / size)
    return np.clip(base * (1-haze[..., None]) + haze[..., None], 0, 1)


def normalize_map(t):
    a = t.detach().cpu().numpy()[0]
    a = np.sqrt(np.mean(a * a, axis=0))
    low, high = np.percentile(a, [2, 98])
    return np.clip((a-low)/(high-low+1e-8), 0, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, default=None, help='Path to your aerial image')
    parser.add_argument('--size', type=int, default=256)
    args = parser.parse_args()
    torch.manual_seed(42)
    torch.set_num_threads(2)
    if args.image:
        image = Image.open(args.image).convert('RGB').resize((args.size, args.size))
        rgb = np.asarray(image, dtype=np.float32) / 255.0
    else:
        rgb = demo_image(args.size)
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
    model = MDFF(channels=3).eval()
    with torch.no_grad():
        tensor = tensor.float()
        h, v, d, fused = model(tensor)
    fig, axes = plt.subplots(1, 5, figsize=(16, 3.8))
    axes[0].imshow(rgb)
    axes[0].set_title('Input image')
    for ax, data, title in zip(axes[1:], [h, v, d, fused], ['1x5 horizontal', '5x1 vertical', '3x3 dilated (d=3)', '1x1 fused']):
        ax.imshow(normalize_map(data), cmap='gray', vmin=0, vmax=1)
        ax.set_title(title)
    for ax in axes:
        ax.axis('off')
    fig.suptitle('Untrained MDFF feature maps (not a dehazed image)', fontsize=13)
    fig.tight_layout()
    out = Path('mdff_feature_maps.png')
    fig.savefig(out, dpi=160, bbox_inches='tight')
    plt.close(fig)
    print('Input shape:', tuple(tensor.shape))
    print('Branch shapes:', tuple(h.shape), tuple(v.shape), tuple(d.shape))
    print('Fused shape:', tuple(fused.shape))
    print('Saved:', out.resolve())
    print('Note: random, untrained convolution weights; maps do not represent learned haze detection.')

if __name__ == '__main__':
    main()
