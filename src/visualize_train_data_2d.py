import random
import argparse

import numpy as np
import matplotlib.pyplot as plt

from utils import read_config
from data_generator_imagine import FetalDataLoader


def visualize_train_data(cfg_path):
    # =========================
    # Load config & data
    # =========================
    config = read_config(cfg_path, mode="train")

    fetal_data = FetalDataLoader(config, Train=True)
    train_dataloader = fetal_data.load_data()

    # =========================
    # randomly take one batch
    # =========================
    batch = next(iter(train_dataloader))

    print(batch.keys())

    images = batch["image"]   # torch.Tensor
    labels = batch["label"]

    B = images.shape[0]
    assert B >= 4, "Batch size must be >= 4"

    indices = random.sample(range(B), 4)

    # =========================
    # plot
    # =========================
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    for col, idx in enumerate(indices):
        img = images[idx].cpu().numpy()
        lab = labels[idx].cpu().numpy()

        # -------------------------
        # process image (2D)
        # -------------------------
        # allowed:
        # (C, H, W) or (H, W)
        if img.ndim == 3:      # (C, H, W)
            img = img[0]

        assert img.ndim == 2, f"Unexpected image shape: {img.shape}"

        # -------------------------
        # process label (2D)
        # -------------------------
        if lab.ndim == 3:      # (C, H, W)
            lab = lab[0]

        assert lab.ndim == 2, f"Unexpected label shape: {lab.shape}"

        # -------------------------
        # debug info
        # -------------------------
        print(f"[IMG] min={img.min():.4f}, max={img.max():.4f}")

        img_path = batch["image_meta_dict"]["filename_or_obj"][idx]
        lab_path = batch["label_meta_dict"]["filename_or_obj"][idx]
        print(f"[VIS] image: {img_path}")
        print(f"[VIS] label: {lab_path}")

        axes[0, col].imshow(img, cmap="gray")
        axes[0, col].set_title(f"Image {idx}")
        axes[0, col].axis("off")

        axes[1, col].imshow(lab, cmap="gray")
        axes[1, col].set_title(f"Label {idx}")
        axes[1, col].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--cfg",
        type=str,
        default="./config_imagine_2D.yml",
        help="path to config file"
    )

    args = parser.parse_args()
    visualize_train_data(args.cfg)
