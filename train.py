
import argparse
import random
import shutil
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch.nn.functional as F

from model import DehazingNet


class PairedDataset(Dataset):

    def __init__(self, root, size=128):
        self.root = Path(root)
        self.size = size

        # Support both folder naming conventions
        hazy_dir = self.find_folder(
            ["hazy", "input"]
        )
        clear_dir = self.find_folder(
            ["clear", "target"]
        )

        if not hazy_dir or not clear_dir:
            raise FileNotFoundError(
                f"Input/target folders not found in {root}"
            )

        extensions = {".png", ".jpg", ".jpeg"}

        def get_key(path):
            # Match names such as 1-inputs.png with 1-targets.png.
            # Also treat numeric names such as 001 and 1 as the same ID.
            stem = path.stem.lower()

            for suffix in ("-inputs", "-targets", "_inputs", "_targets"):
                if stem.endswith(suffix):
                    stem = stem[:-len(suffix)]
                    break

            if stem.isdigit():
                stem = str(int(stem))

            return stem

        def build_file_map(folder, folder_label):
            file_map = {}

            for path in folder.iterdir():
                if not path.is_file() or path.suffix.lower() not in extensions:
                    continue

                key = get_key(path)

                if key in file_map:
                    raise ValueError(
                        f"Duplicate {folder_label} image ID '{key}' in {folder}: "
                        f"{file_map[key].name} and {path.name}"
                    )

                file_map[key] = path

            return file_map

        hazy_files = build_file_map(hazy_dir, "hazy")
        clear_files = build_file_map(clear_dir, "clear")

        common = sorted(
            set(hazy_files) & set(clear_files),
            key=lambda value: (not value.isdigit(), int(value) if value.isdigit() else value)
        )

        unmatched_hazy = sorted(set(hazy_files) - set(clear_files))
        unmatched_clear = sorted(set(clear_files) - set(hazy_files))

        print(
            f"Matched {len(common)} pairs | "
            f"Unmatched hazy: {len(unmatched_hazy)} | "
            f"Unmatched clear: {len(unmatched_clear)}"
        )

        self.pairs = [
            (hazy_files[name], clear_files[name])
            for name in common
        ]

        if not self.pairs:
            hazy_examples = [p.name for p in list(hazy_dir.iterdir())[:5] if p.is_file()]
            clear_examples = [p.name for p in list(clear_dir.iterdir())[:5] if p.is_file()]
            raise ValueError(
                f"No matching image pairs found in {root}. "
                f"Example hazy files: {hazy_examples}; "
                f"example clear files: {clear_examples}"
            )

        print(
            f"{self.root}: {len(self.pairs)} image pairs"
        )

    def find_folder(self, names):
        for name in names:
            folder = self.root / name
            if folder.is_dir():
                return folder
        return None

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):

        hazy_path, clear_path = self.pairs[index]

        def load(path):
            with Image.open(path) as image:
                image = image.convert("RGB")
                image = image.resize(
                    (self.size, self.size),
                    Image.Resampling.BICUBIC
                )

                array = np.asarray(
                    image, dtype=np.float32
                ).copy() / 255.0

            return torch.from_numpy(
                array
            ).permute(2, 0, 1)

        return load(hazy_path), load(clear_path)


def calculate_metrics(prediction, target):

    prediction = prediction.clamp(0, 1)
    target = target.clamp(0, 1)

    mse = torch.mean(
        (prediction - target) ** 2,
        dim=(1, 2, 3)
    )

    psnr = -10 * torch.log10(
        mse.clamp_min(1e-10)
    )

    # SSIM calculation
    mu1 = F.avg_pool2d(
        prediction, 11, stride=1, padding=5,
        count_include_pad=False
    )

    mu2 = F.avg_pool2d(
        target, 11, stride=1, padding=5,
        count_include_pad=False
    )

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu12 = mu1 * mu2

    sigma1_sq = (
        F.avg_pool2d(
            prediction ** 2, 11, 1, 5,
            count_include_pad=False
        ) - mu1_sq
    )

    sigma2_sq = (
        F.avg_pool2d(
            target ** 2, 11, 1, 5,
            count_include_pad=False
        ) - mu2_sq
    )

    sigma12 = (
        F.avg_pool2d(
            prediction * target, 11, 1, 5,
            count_include_pad=False
        ) - mu12
    )

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2

    ssim_map = (
        (2 * mu12 + c1) *
        (2 * sigma12 + c2)
    ) / (
        (mu1_sq + mu2_sq + c1) *
        (sigma1_sq + sigma2_sq + c2)
    )

    ssim = ssim_map.mean(dim=(1, 2, 3))

    mae = torch.mean(
        torch.abs(prediction - target),
        dim=(1, 2, 3)
    )

    return psnr, ssim, mae


def save_comparison(hazy, prediction, clear, path):

    images = [
        hazy, prediction, clear
    ]

    titles = [
        "Hazy Input",
        "Dehazed Output",
        "Ground Truth"
    ]

    plt.figure(figsize=(12, 4))

    for i, (image, title) in enumerate(
        zip(images, titles)
    ):
        image = image.detach().cpu()
        image = image.permute(1, 2, 0).numpy()
        image = np.clip(image, 0, 1)

        plt.subplot(1, 3, i + 1)
        plt.imshow(image)
        plt.title(title)
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def evaluate_loss(model, loader, criterion, device):

    model.eval()
    total_loss = 0.0

    with torch.no_grad():

        for hazy, clear in loader:

            hazy = hazy.to(device)
            clear = clear.to(device)

            prediction = model(hazy)

            loss = criterion(prediction, clear)

            total_loss += (
                loss.item() * hazy.size(0)
            )

    return total_loss / len(loader.dataset)


def test_model(model, loader, device, output_dir):

    model.eval()

    total_psnr = 0.0
    total_ssim = 0.0
    total_mae = 0.0
    count = 0

    comparison_dir = output_dir / "test_comparisons"
    comparison_dir.mkdir(exist_ok=True)

    with torch.no_grad():

        for batch_index, (hazy, clear) in enumerate(loader):

            hazy = hazy.to(device)
            clear = clear.to(device)

            prediction = model(hazy)

            psnr, ssim, mae = calculate_metrics(
                prediction, clear
            )

            total_psnr += psnr.sum().item()
            total_ssim += ssim.sum().item()
            total_mae += mae.sum().item()

            count += hazy.size(0)

            # Save first 10 comparison images
            for i in range(hazy.size(0)):

                image_number = batch_index * loader.batch_size + i

                if image_number < 10:

                    save_comparison(
                        hazy[i],
                        prediction[i],
                        clear[i],
                        comparison_dir /
                        f"comparison_{image_number + 1}.png"
                    )

    if count == 0:
        raise ValueError("Test dataset is empty.")

    avg_psnr = total_psnr / count
    avg_ssim = total_ssim / count
    avg_mae = total_mae / count

    print("\n========== TEST RESULTS ==========")
    print(f"Test images : {count}")
    print(f"Average PSNR: {avg_psnr:.4f} dB")
    print(f"Average SSIM: {avg_ssim:.4f}")
    print(f"Average MAE : {avg_mae:.6f}")
    print("==================================")

    with open(output_dir / "test_results.txt", "w") as f:
        f.write(f"Test images: {count}\n")
        f.write(f"PSNR: {avg_psnr:.4f} dB\n")
        f.write(f"SSIM: {avg_ssim:.4f}\n")
        f.write(f"MAE: {avg_mae:.6f}\n")


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data", default="dataset"
    )

    parser.add_argument(
        "--epochs", type=int, default=30
    )

    parser.add_argument(
        "--batch", type=int, default=2
    )

    parser.add_argument(
        "--size", type=int, default=128
    )

    parser.add_argument(
        "--lr", type=float, default=0.0001
    )

    parser.add_argument(
        "--start-epoch", type=int, default=30,
        help="Completed epochs in the existing checkpoint"
    )

    args = parser.parse_args()

    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)

    torch.set_num_threads(2)

    device = torch.device("cpu")

    print("Training device:", device)

    data_root = Path(args.data)

    train_dataset = PairedDataset(
        data_root / "train",
        args.size
    )

    # Accept either validation or val
    if (data_root / "validation").is_dir():
        validation_path = data_root / "validation"
    else:
        validation_path = data_root / "val"

    val_dataset = PairedDataset(
        validation_path,
        args.size
    )

    test_dataset = PairedDataset(
        data_root / "test",
        args.size
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch,
        shuffle=False,
        num_workers=0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch,
        shuffle=False,
        num_workers=0
    )

    print("\nDataset summary")
    print("Training pairs:", len(train_dataset))
    print("Validation pairs:", len(val_dataset))
    print("Testing pairs:", len(test_dataset))

    model = DehazingNet(
        base_channels=16
    ).to(device)

    criterion = nn.L1Loss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-7
    )

    output_dir = (
        Path(__file__).resolve().parent / "outputs"
    )

    output_dir.mkdir(exist_ok=True)

    train_losses = []
    val_losses = []

    best_path = output_dir / "best_model.pth"
    last_path = output_dir / "last_model.pth"

    if not best_path.is_file() or not last_path.is_file():
        raise FileNotFoundError(
            "Both outputs/best_model.pth and outputs/last_model.pth "
            "must exist before resuming."
        )

    # Keep untouched copies of the checkpoints from the first 30 epochs.
    best_backup = output_dir / "best_model_before_resume.pth"
    last_backup = output_dir / "last_model_before_resume.pth"

    if not best_backup.exists():
        shutil.copy2(best_path, best_backup)
    if not last_backup.exists():
        shutil.copy2(last_path, last_backup)

    print("\\nLoading epoch-30 latest checkpoint...")
    model.load_state_dict(
        torch.load(last_path, map_location=device, weights_only=True)
    )

    # Establish the old best validation score so a worse new model
    # cannot overwrite the original best checkpoint.
    print("Checking the previous best checkpoint...")
    model.load_state_dict(
        torch.load(best_path, map_location=device, weights_only=True)
    )
    best_loss = evaluate_loss(model, val_loader, criterion, device)
    print(f"Previous best validation L1: {best_loss:.6f}")

    # Continue optimization from the latest (epoch-30) weights.
    model.load_state_dict(
        torch.load(last_path, map_location=device, weights_only=True)
    )

    print(
        f"\\nContinuing from epoch {args.start_epoch} "
        f"for {args.epochs} more epochs."
    )
    print(
        "Note: model weights are resumed; Adam optimizer state starts fresh."
    )

    # ================= TRAINING =================

    for epoch in range(args.epochs):

        model.train()

        running_loss = 0.0

        for hazy, clear in train_loader:

            hazy = hazy.to(device)
            clear = clear.to(device)

            optimizer.zero_grad()

            prediction = model(hazy)

            loss = criterion(
                prediction, clear
            )

            loss.backward()

            optimizer.step()

            running_loss += (
                loss.item() * hazy.size(0)
            )

        train_loss = (
            running_loss / len(train_dataset)
        )

        # Validation after every epoch
        val_loss = evaluate_loss(
            model,
            val_loader,
            criterion,
            device
        )

        scheduler.step()

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(
            f"Epoch {args.start_epoch + epoch + 1}/"
            f"{args.start_epoch + args.epochs} | "
            f"Train L1: {train_loss:.6f} | "
            f"Val L1: {val_loss:.6f}",
            flush=True
        )

        # Save best model
        if val_loss < best_loss:

            best_loss = val_loss

            torch.save(
                model.state_dict(),
                output_dir / "best_model.pth"
            )

            print("Best model saved.")

        # Save latest model
        torch.save(
            model.state_dict(),
            output_dir / "last_model.pth"
        )

    # ================= LOSS GRAPH =================

    plt.figure(figsize=(8, 5))

    epoch_numbers = list(
        range(args.start_epoch + 1, args.start_epoch + args.epochs + 1)
    )

    plt.plot(
        epoch_numbers, train_losses, label="Training L1"
    )

    plt.plot(
        epoch_numbers, val_losses, label="Validation L1"
    )

    plt.xlabel("Epoch")
    plt.ylabel("L1 Loss")
    plt.title("MDFF Dehazing - Continued Training")

    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        output_dir / "training_loss.png",
        dpi=160
    )

    plt.close()

    # ================= FINAL TESTING =================

    print("\nLoading best model for testing...")

    model.load_state_dict(
        torch.load(
            output_dir / "best_model.pth",
            map_location=device,
            weights_only=True
        )
    )

    test_model(
        model,
        test_loader,
        device,
        output_dir
    )

    print("\nTraining and testing completed.")
    print("Best validation L1:", best_loss)
    print("Results saved in:", output_dir)


if __name__ == "__main__":
    main()
