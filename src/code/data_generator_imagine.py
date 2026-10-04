import os

import numpy as np
import pandas as pd
from glob import glob

import monai.transforms as tr
from monai.data import Dataset, CacheDataset, DataLoader, ThreadDataLoader


from utils import SliceWiseNormalizeIntensityd

import warnings

warnings.filterwarnings('ignore')


class FetalDataLoader:
    """Data loader for fetal imaging segmentation, handling both 2D and 3D data."""

    def __init__(self, config, Train=True):
        self.config = config
        self.Train = Train

    def train_transformations_2d(self):
        train_trans = [
            tr.LoadImaged(keys=["image", "label"], image_only=False),
            tr.EnsureChannelFirstd(keys=["image", "label"]),
            tr.Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, -1), mode=("bilinear", "nearest")),
            tr.RandSpatialCropd(
                keys=["image", "label"],
                roi_size=(self.config.img_size[0], self.config.img_size[1]),
                random_center=True,
                random_size=False,
            ),
            tr.ResizeWithPadOrCropd(
                keys=["image", "label"],
                spatial_size=(
                    self.config.img_size[0],
                    self.config.img_size[1],
                ),
                method="symmetric",
                mode="constant",
            ),
        ]

        if self.config.augmentation:
            spatial_aug = tr.OneOf([
                tr.RandFlipd(keys=["image", "label"], spatial_axis=0, prob=1),
                tr.RandFlipd(keys=["image", "label"], spatial_axis=1, prob=1),
                tr.RandRotate90d(
                    keys=["image", "label"],
                    spatial_axes=(0, 1),
                    prob=1,
                ),
                tr.RandRotated(
                    keys=["image", "label"],
                    range_x=(0.2, 1.0),
                    prob=1,
                    mode = ("bilinear", "nearest")
                )
            ])

            intensity_aug = tr.OneOf([
                tr.RandRicianNoised(keys=["image"], prob=0.7),
                tr.RandBiasFieldd(keys=["image"], degree=4, coeff_range=(0.05, 0.15), prob=0.7),
            ])

            train_trans.append(spatial_aug)
            train_trans.append(intensity_aug)

            train_trans.extend([
                tr.ScaleIntensityRangePercentilesd(
                    keys=["image"],
                    lower=1,
                    upper=99,
                    b_min=0.0,
                    b_max=1.0,
                    clip=True
                ),
                tr.RandScaleIntensityd(
                    keys=["image"],
                    factors=(-0.5, -0),
                    prob=0.3,
                ),
            ])

        return tr.Compose(train_trans)

    def train_transformations_3d(self):
        train_trans = [
            tr.LoadImaged(keys=["image", "label"],image_only=False),
            tr.EnsureChannelFirstd(keys=["image", "label"]),
            # spacing for x,y only
            tr.Spacingd(
                keys=["image", "label"],
                pixdim=(
                    self.config.voxel_spacing[0],
                    self.config.voxel_spacing[1],
                    -1
                ),
                mode=("bilinear", "nearest"),
            ),
            # spacing for z only (nearest)
            tr.Spacingd(
                keys=["image", "label"],
                pixdim=(
                    -1,
                    -1,
                    self.config.voxel_spacing[2],
                ),
                mode=("nearest", "nearest"),
            ),
            tr.RandSpatialCropD(
                keys=["image", "label"],
                roi_size=(
                    self.config.img_size[0],
                    self.config.img_size[1],
                    self.config.img_size[2]
                ),
                random_center=True,
                random_size=False
            ),
            tr.ResizeWithPadOrCropd(
                keys=["image", "label"],
                spatial_size=(
                    self.config.img_size[0],
                    self.config.img_size[1],
                    self.config.img_size[2]
                ),
                method="symmetric",
                mode="constant",
            ),
        ]

        if self.config.augmentation:
            spatial_aug = tr.OneOf([
                tr.RandFlipd(keys=["image", "label"], spatial_axis=1, prob=0.6),
                tr.RandRotate90d(keys=["image", "label"], spatial_axes=(0, 1), prob=0.6),
            ])

            intensity_aug = tr.OneOf([
                tr.RandRicianNoised(keys=["image"], prob=0.6),
                tr.RandBiasFieldd(keys=["image"], degree=4, coeff_range=(0.05, 0.15), prob=0.6),
                tr.RandGaussianSmoothd(keys=["image"], sigma_x=(0.5, 1.0), sigma_y=(0.5, 1.0), prob=0.6),
            ])

            train_trans.extend([spatial_aug, intensity_aug])

        train_trans.extend([
            tr.ScaleIntensityRangePercentilesd(
                keys=["image"],
                lower=1,
                upper=99,
                b_min=0.0,
                b_max=1.0,
                clip=True
            ),
            tr.RandScaleIntensityd(
                keys=["image"],
                factors=(-0.5,-0),
                prob=0.3,
            ),
        ])
        return tr.Compose(train_trans)

    def test_transformations_2d(self):
        test_transforms_list = [
            tr.LoadImaged(keys=["image", "label"]),
            tr.EnsureChannelFirstd(keys=["image", "label"]),
            tr.Spacingd(keys="image", pixdim=(1.0, 1.0, -1.0), mode="bilinear"),
            SliceWiseNormalizeIntensityd(keys=["image"], subtrahend=0.0, divisor=None, nonzero=True),
        ]
        return test_transforms_list

    def test_transformations_3d(self):
        test_transforms_list = [
            tr.LoadImaged(
                keys=["image"],
                image_only=False
            ),
            tr.EnsureChannelFirstd(
                keys=["image"]
            ),
            # spacing for x,y only
            tr.Spacingd(
                keys=["image"],
                pixdim=(
                    self.config.voxel_spacing[0],
                    self.config.voxel_spacing[1],
                    -1
                ),
                mode=("bilinear", "nearest"),
            ),
            # spacing for z only (nearest)
            tr.Spacingd(
                keys=["image"],
                pixdim=(
                    -1,
                    -1,
                    self.config.voxel_spacing[2],
                ),
                mode=("nearest", "nearest"),
            ),
            tr.RandSpatialCropD(
                keys=["image"],
                roi_size=(
                    self.config.img_size[0],
                    self.config.img_size[1],
                    self.config.img_size[2]
                ),
                random_center=True,
                random_size=False
            ),
            tr.ResizeWithPadOrCropd(
                keys=["image", "label"],
                spatial_size=(
                    self.config.img_size[0],
                    self.config.img_size[1],
                    self.config.img_size[2]
                ),
                method="symmetric",
                mode="constant",
            ),
            tr.ScaleIntensityRangePercentilesd(
                keys=["image"],
                lower=1,
                upper=99,
                b_min=0.0,
                b_max=1.0,
                clip=True
            ),
        ]
        return test_transforms_list

    def load_data(self):
        """Build dataloaders from config. Train=True returns train_dataloader; Train=False
        returns (test_dataloader, test_files, test_transforms_list)."""
        if self.Train:
            if self.config.spatial_dims == "2d":
                train_transforms= self.train_transformations_2d()
            elif self.config.spatial_dims == "3d":
                train_transforms= self.train_transformations_3d()

            if self.config.train_data_type == "path":
                train_images = sorted(glob(os.path.join(self.config.train_data_paths, 'images', "img_*.nii.gz")))
                train_labels = sorted(glob(os.path.join(self.config.train_data_paths, 'masks', "mask_*.nii.gz")))
            elif self.config.train_data_type == "file":
                train_images = []
                train_labels = []
                for path in self.config.train_data_paths:
                    data = pd.read_csv(path)
                    train_images += data.iloc[:, 0].tolist()
                    train_labels += data.iloc[:, 1].tolist()

            train_files = [{"image": image_name, "label": label_name} for
                           image_name, label_name in zip(train_images, train_labels)]

            if self.config.fast_training:
                train_dataset = CacheDataset(data=train_files,
                                             transform=train_transforms,
                                             cache_rate=1.0,
                                             num_workers=8,
                                             copy_cache=False)

                train_dataloader = ThreadDataLoader(train_dataset,
                                                    num_workers=0,
                                                    batch_size=self.config.batch_size,
                                                    shuffle=True)
            else:
                train_dataset = Dataset(data=train_files,
                                        transform=train_transforms)

                train_dataloader = DataLoader(train_dataset,
                                              batch_size=self.config.batch_size,
                                              shuffle=True,
                                              num_workers=8,
                                              pin_memory=False)

            return train_dataloader

        else:
            if self.config.test_data_type == "path":
                test_images = sorted(glob(os.path.join(self.config.test_data_paths, 'data', '**/*.nii.gz'),
                                          recursive=True))
                test_labels = sorted(glob(os.path.join(self.config.test_data_paths, 'manual-masks', '**/*.nii.gz'),
                                          recursive=True))
            elif self.config.test_data_type == "file":
                test_images = []
                test_labels = []
                for path in self.config.test_data_paths:
                    data = pd.read_csv(path)
                    test_images += data.iloc[:, 0].tolist()
                    test_labels += data.iloc[:, 1].tolist()

            test_files = [{"image": image_name, "label": label_name} for
                          image_name, label_name in zip(test_images, test_labels)]

            if self.config.spatial_dims == "2d":
                test_transforms_list = self.test_transformations_2d()
            else:
                test_transforms_list = self.test_transformations_3d()

            test_dataset = Dataset(data=test_files, transform=tr.Compose(test_transforms_list))
            test_dataloader = DataLoader(test_dataset,
                                         batch_size=1,
                                         shuffle=False,
                                         num_workers=0)

            return test_dataloader, test_files, test_transforms_list
