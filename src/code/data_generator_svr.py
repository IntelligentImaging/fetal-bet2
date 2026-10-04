"""
Data loading for the SVR-reconstructed brain volumes (volume_svr / volume_svr_mask).

Kept separate from data_generator_imagine.py (which is shared by other configs, e.g.
config_imagine.yml) so SVR-specific augmentation choices don't affect other pipelines.
"""

import os

import numpy as np
import pandas as pd

import monai.transforms as tr
from monai.data import Dataset, DataLoader
from monai.transforms import MapTransform


class ConditionalTransformd(MapTransform):
    """Apply `transform` only when the sample's image filename starts with `prefix` -
    used to restrict spatial augmentation to the SVR data (filenames like "SUB-...")."""

    def __init__(self, transform, prefix="SUB", image_key="image"):
        self.transform = transform
        self.prefix = prefix
        self.image_key = image_key

    def __call__(self, data):
        filename = data.get(f"{self.image_key}_meta_dict", {}).get("filename_or_obj")
        if filename is not None and os.path.basename(filename).startswith(self.prefix):
            return self.transform(data)
        return dict(data)


def _load_file_pairs(csv_paths):
    images, labels = [], []
    for path in csv_paths:
        data = pd.read_csv(path)
        images += data.iloc[:, 0].tolist()
        labels += data.iloc[:, 1].tolist()
    return [{"image": i, "label": l} for i, l in zip(images, labels)]


def build_train_transforms(config):
    voxel_spacing = config.voxel_spacing
    img_size = config.img_size

    train_trans = [
        tr.LoadImaged(keys=["image", "label"], image_only=False),
        tr.EnsureChannelFirstd(keys=["image", "label"]),
        # spacing for x,y only
        tr.Spacingd(
            keys=["image", "label"],
            pixdim=(voxel_spacing[0], voxel_spacing[1], -1),
            mode=("bilinear", "nearest"),
        ),
        # spacing for z only (nearest)
        tr.Spacingd(
            keys=["image", "label"],
            pixdim=(-1, -1, voxel_spacing[2]),
            mode=("nearest", "nearest"),
        ),
        tr.RandSpatialCropD(
            keys=["image", "label"],
            roi_size=(img_size[0], img_size[1], img_size[2]),
            random_center=True,
            random_size=False,
        ),
        tr.ResizeWithPadOrCropd(
            keys=["image", "label"],
            spatial_size=(img_size[0], img_size[1], img_size[2]),
            method="symmetric",
            mode="constant",
        ),
    ]

    if config.augmentation:
        # flip / 90-degree rotation (pick one)
        spatial_aug = tr.OneOf([
            tr.RandFlipd(keys=["image", "label"], spatial_axis=1, prob=0.6),
            tr.RandRotate90d(keys=["image", "label"], spatial_axes=(0, 1), prob=0.6),
        ])

        # full-range (+/- 180 deg) rotation on all three axes
        rotate_aug = tr.RandRotated(
            keys=["image", "label"],
            range_x=(-np.pi, np.pi),
            range_y=(-np.pi, np.pi),
            range_z=(-np.pi, np.pi),
            prob=0.5,
            mode=("bilinear", "nearest"),
            padding_mode="zeros",
            keep_size=True,
        )

        # random resolution scaling 0.7x - 1.3x, cropped/padded back to img_size
        zoom_aug = tr.RandZoomd(
            keys=["image", "label"],
            min_zoom=0.7,
            max_zoom=1.3,
            mode=("trilinear", "nearest"),
            padding_mode="constant",
            prob=0.5,
            keep_size=True,
        )

        # bias field + smoothing, before percentile normalization absorbs any edge extremes
        train_trans.extend([
            tr.RandBiasFieldd(
                keys=["image"],
                degree=2,
                coeff_range=(0.3, 0.6),
                prob=0.5,
            ),
            tr.RandGaussianSmoothd(
                keys=["image"],
                sigma_x=(0.5, 1.5),
                sigma_y=(0.5, 1.5),
                sigma_z=(0.5, 1.5),
                prob=0.4,
            ),
        ])

        # spatial augmentation is restricted to SVR data (filenames starting with "SUB")
        spatial_aug_combo = tr.Compose([spatial_aug, rotate_aug, zoom_aug])
        train_trans.append(ConditionalTransformd(spatial_aug_combo, prefix="SUB"))

    train_trans.append(
        tr.ScaleIntensityRangePercentilesd(
            keys=["image"],
            lower=1,
            upper=99,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        )
    )

    if config.augmentation:
        # noise added after normalization, so its magnitude is calibrated to [0, 1]
        train_trans.extend([
            tr.RandRicianNoised(
                keys=["image"],
                prob=0.5,
                std=0.15,
                sample_std=True,
            ),
            # clip the (rare) noise overshoot back into [0, 1]
            tr.ScaleIntensityRanged(
                keys=["image"], a_min=0.0, a_max=1.0, b_min=0.0, b_max=1.0, clip=True,
            ),
            tr.RandScaleIntensityd(
                keys=["image"],
                factors=(-0.5, -0),
                prob=0.3,
            ),
        ])

    return tr.Compose(train_trans)


def build_test_transforms(config):
    voxel_spacing = config.voxel_spacing
    return tr.Compose([
        tr.LoadImaged(keys=["image", "label"], image_only=False),
        tr.EnsureChannelFirstd(keys=["image", "label"]),
        tr.Spacingd(
            keys=["image", "label"],
            pixdim=(voxel_spacing[0], voxel_spacing[1], voxel_spacing[2]),
            mode=("bilinear", "nearest"),
        ),
        tr.ScaleIntensityRangePercentilesd(
            keys=["image"],
            lower=1,
            upper=99,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
    ])


def load_train_data(config, num_workers=8):
    """Patch-based, augmented training dataloader."""
    train_files = _load_file_pairs(config.train_data_paths)
    train_dataset = Dataset(data=train_files, transform=build_train_transforms(config))
    return DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
    )


def load_test_data(config, num_workers=4):
    """Full-volume test dataloader (image + label kept, for loss/Dice tracking)."""
    test_files = _load_file_pairs(config.test_data_paths)
    test_dataset = Dataset(data=test_files, transform=build_test_transforms(config))
    return DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=num_workers)
