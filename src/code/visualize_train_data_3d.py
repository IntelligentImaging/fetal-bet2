import random
import argparse

import numpy as np
import matplotlib.pyplot as plt

from utils import read_config
from data_generator_imagine import FetalDataLoader


def visualize_train_data(cfg_path):
    # Load config & data
    config = read_config(cfg_path, mode="train")

    fetal_data = FetalDataLoader(config, Train=True)
    train_dataloader = fetal_data.load_data()

    # randomly take one batch
    batch = next(iter(train_dataloader))

    print(batch.keys())

    images = batch["image"]   # torch.Tensor
    labels = batch["label"]


    B = images.shape[0]
    assert B >= 4, "Batch size must be >= 4"

    indices = random.sample(range(B), 4)

    # plot
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    for col, idx in enumerate(indices):
        img = images[idx].cpu().numpy()
        lab = labels[idx].cpu().numpy()

        # img/lab shape may be (C, H, W, D) or (C, H, W) or (H, W, D)
        if img.ndim == 4:
            img = img[0]

        if img.ndim == 3:
            z = img.shape[-1] // 2
            img = img[:, :, z]

        if lab.ndim == 4:
            lab = lab[0]

        if lab.ndim == 3:
            z = lab.shape[-1] // 2
            lab = lab[:, :, z]

        print(img.min())
        print(img.max())

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
        default="./config_imagine.yml",
        help="path to config file"
    )

    args = parser.parse_args()
    visualize_train_data(args.cfg)
