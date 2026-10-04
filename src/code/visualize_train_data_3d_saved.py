import os
import random
import argparse

import numpy as np
import matplotlib.pyplot as plt

CODE_DIR = os.path.dirname(os.path.abspath(__file__))

from utils import read_config
from data_generator_svr import load_train_data


def visualize_train_data(cfg_path, save_path=None):
    # Load config & data
    config = read_config(cfg_path, mode="train")

    train_dataloader = load_train_data(config)

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

    path_lines = []

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
        path_lines.append(f"Image {idx}: {img_path}")
        path_lines.append(f"Label {idx}: {lab_path}")

        axes[0, col].imshow(img, cmap="gray")
        axes[0, col].set_title(f"Image {idx}")
        axes[0, col].axis("off")

        axes[1, col].imshow(lab, cmap="gray")
        axes[1, col].set_title(f"Label {idx}")
        axes[1, col].axis("off")

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150)
        print(f"[VIS] figure saved to: {save_path}")

        paths_txt = os.path.splitext(save_path)[0] + "_paths.txt"
        with open(paths_txt, "w") as f:
            f.write("\n".join(path_lines) + "\n")
        print(f"[VIS] file paths saved to: {paths_txt}")

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--cfg",
        type=str,
        default=os.path.join(CODE_DIR, "config_inference_3d.yml"),
        help="path to config file"
    )

    parser.add_argument(
        "--save_path",
        type=str,
        default=None,
        help="if set, also save the figure to this path (useful when there is no display)"
    )

    args = parser.parse_args()
    visualize_train_data(args.cfg, args.save_path)
