import numpy as np
import yaml

from monai.transforms import MapTransform

import warnings

warnings.filterwarnings('ignore')


class Config:
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            setattr(self, key, value)


def read_config(file_path, mode='train'):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)

    general_config = {k: v for k, v in config.items() if (k != 'train_mode' and k != 'test_mode')}

    if mode == 'train':
        train_config = config['train_mode']
        combined_config = {**general_config, **train_config}
    else:
        train_config = config['test_mode']
        combined_config = {**general_config, **train_config}

    return Config(combined_config)


class SliceWiseNormalizeIntensityd(MapTransform):
    def __init__(self, keys, subtrahend=0.0, divisor=None, nonzero=True):
        super().__init__(keys)
        self.subtrahend = subtrahend
        self.divisor = divisor
        self.nonzero = nonzero

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            image = d[key]
            for i in range(image.shape[-1]):
                slice_ = image[..., i]
                if self.nonzero:
                    mask = slice_ > 0
                    if np.any(mask):
                        if self.subtrahend is None:
                            slice_[mask] = slice_[mask] - slice_[mask].mean()
                        else:
                            slice_[mask] = slice_[mask] - self.subtrahend

                        if self.divisor is None:
                            slice_[mask] /= slice_[mask].std()
                        else:
                            slice_[mask] /= self.divisor

                else:
                    if self.subtrahend is None:
                        slice_ = slice_ - slice_.mean()
                    else:
                        slice_ = slice_ - self.subtrahend

                    if self.divisor is None:
                        slice_ /= slice_.std()
                    else:
                        slice_ /= self.divisor

                image[..., i] = slice_
            d[key] = image
        return d
