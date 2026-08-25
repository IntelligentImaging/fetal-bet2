import os
from glob import glob
from pathlib import Path

import argparse

import torch

PROJECT_DIR = Path(__file__).resolve().parent.parent

import monai.transforms as tr
from monai.data import decollate_batch, Dataset, DataLoader
from monai.inferers import SlidingWindowInferer
from monai.networks.nets import AttentionUnet
from monai.transforms import SaveImaged
from monai.utils import set_determinism


class FetalTestData:
    def __init__(self, test_data_paths, img_size=256):
        self.test_data_paths = test_data_paths
        self.img_size = img_size

    def transformations(self):
        test_transforms_list = [
            tr.LoadImaged(
                keys=["image"],
            ),
            tr.EnsureChannelFirstd(
                keys=["image"]
            ),
            tr.Spacingd(
                keys=["image"],
                pixdim=(
                    1.0,
                    1.0,
                    -1
                ),
                mode="bilinear",padding_mode="zeros"
            ),
            tr.Spacingd(
                keys=["image"],
                pixdim=(
                    -1,
                    -1,
                    1.0,
                ),
                mode="nearest",padding_mode="zeros"
            ),
            tr.ScaleIntensityRangePercentilesd(
                keys=["image"],
                lower=1,
                upper=99,
                b_min=0.0,
                b_max=1,
                clip=True
            ),
        ]
        return test_transforms_list

    def load_data(self):
        test_transforms_list = self.transformations()
        test_images = sorted(glob(os.path.join(self.test_data_paths, "*.nii.gz")))

        test_files = [{"image": image_name} for image_name in test_images]

        test_dataset = Dataset(data=test_files, transform=tr.Compose(test_transforms_list))
        test_dataloader = DataLoader(test_dataset,
                                     batch_size=1,
                                     num_workers=0)

        return test_dataloader, test_transforms_list


def inference(args):
    model = AttentionUnet(
    spatial_dims=3,
    in_channels=1,
    out_channels=2,
    channels=(32, 64, 128, 256, 512),
    strides=(2, 2, 2, 2),
    kernel_size=3,
    up_kernel_size=3,
    dropout=0
    )

    device = args.device
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
        model.to(device)
    else:
        model.to(device)

    model.load_state_dict(torch.load(args.saved_model_path, map_location=device))
    model.eval()

    fetal_test_data = FetalTestData(args.data_path)
    test_dataloader, test_org_transforms_list = fetal_test_data.load_data()

    inferer = SlidingWindowInferer(roi_size=(224, 224, 128), sw_batch_size=1, overlap = 0.25)

    post_transforms = tr.Compose([
        tr.Invertd(
            keys="pred",
            transform=tr.Compose(test_org_transforms_list),
            orig_keys="image",
            meta_keys="pred_meta_dict",
            orig_meta_keys="image_meta_dict",
            meta_key_postfix="meta_dict",
            nearest_interp=False,
            to_tensor=True,
        ),
        tr.Activationsd(keys="pred", softmax=True),
        tr.AsDiscreted(keys="pred", argmax=True, to_onehot=None),
        SaveImaged(keys="pred", meta_keys="pred_meta_dict", output_dir=args.save_path,
                   separate_folder=False, output_postfix="predicted_mask", resample=False),
    ]
    )

    with torch.no_grad():
        for i, test_data in enumerate(test_dataloader):
            test_inputs = test_data["image"].to(device)
            test_data["pred"] = inferer(test_inputs, model)
            test_data = [post_transforms(i) for i in decollate_batch(test_data)]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--n_gpu',
                        type=int,
                        default=2,
                        help='total gpu')

    parser.add_argument('--deterministic',
                        type=int,
                        default=1,
                        help='whether use deterministic training')

    parser.add_argument('--seed',
                        type=int,
                        default=1234,
                        help='random seed')

    parser.add_argument('--device',
                        type=str,
                        default=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                        help='what device to use')

    parser.add_argument('--saved_model_path',
                        type=str,
                        default=str(PROJECT_DIR / "Docker" / "src" / "models" / "AttUNet3D.pth"),
                        help='path to the saved model')

    parser.add_argument('--data_path',
                        type=str,
                        default=str(PROJECT_DIR / "data_test"),
                        help='path to the test data')

    parser.add_argument('--save_path',
                        type=str,
                        default=str(PROJECT_DIR / "data_test_3D"),
                        help='path to save the out')

    args = parser.parse_args()

    if args.deterministic:
        set_determinism(seed=12345)
    else:
        set_determinism(seed=None)

    inference(args)
