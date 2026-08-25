# Fetal-BET2: Brain Extraction Tool for Fetal MRI

Fetal brain extraction is performed using **Fetal-BET2**, a multimodal fetal brain
extraction model developed by our group and publicly available on
[GitHub](https://github.com/IntelligentImaging/fetal-bet2) and
[Docker Hub](https://github.com/IntelligentImaging/fetal-bet2/pkgs/container/fetal-bet2)
(`ghcr.io/intelligentimaging/fetal-bet2:latest`).

Fetal-BET2 is a fully supervised deep learning method based on a U-Net architecture,
trained with extensive data augmentation to improve robustness to variations in fetal
orientation, image contrast, noise, and acquisition characteristics. The model was
trained on approximately 1,600 3D volumes and 55,000 2D slices across T2w, dMRI, and
fMRI.

Fetal-BET2 supports both 2D and 3D inference:
- **2D inference** is used for **T2w stacks**, to better accommodate interslice fetal
  motion before reconstruction.
- **3D inference** is used for **SVR-reconstructed T2w volumes, dMRI, and fMRI**,
  because of their rapid EPI readout and reduced interslice motion (or, for SVR, because
  motion has already been corrected during reconstruction).

![HASTE and SVR segmentation](./plots/HASTE_SVR_comparison.gif)

![dMRI_b0, dMRI_b900, and fMRI segmentation](./plots/dMRIb0_dMRIb900_fMRI_comparison.gif)

## Table of Contents
- [Installation](#installation)
  - [Requirements](#requirements)
  - [Docker](#docker)
- [Usage](#usage)
  - [Training](#training)
  - [Inference](#inference)
- [Fine-tuning on New Data](#fine-tuning-on-new-data)
- [Contact](#contact)
- [Acknowledgement](#acknowledgement)

## Installation

### Requirements

- Python 3.9
- torch==1.12.1+cu113
- monai==1.2.0
- Docker 20.10+ (optional but recommended for inference)

See `src/requirements.txt` for the full list.

```bash
# Clone the repository
git clone https://github.com/IntelligentImaging/fetal-bet2.git
cd fetal-bet2

# Install Python dependencies
pip install -r src/requirements.txt
```

### Docker

```bash
# Pull the Docker image
docker pull ghcr.io/intelligentimaging/fetal-bet2:latest
```

## Usage

### Training

```bash
cd src
# --cfg selects the model dimensionality: config_imagine.yml (3D) or config_imagine_2D.yml (2D)
python train.py --cfg config_imagine.yml
python train.py --cfg config_imagine_2D.yml
```

### Inference

```bash
cd src

# 2D inference (e.g. T2w stacks)
python inference_2d.py \
  --saved_model_path /path/to/AttUNet.pth \
  --data_path /path/to/input_dir/ \
  --save_path /path/to/output_dir/

# 3D inference (e.g. SVR-reconstructed T2w, dMRI, fMRI)
python inference_3d.py \
  --saved_model_path /path/to/AttUNet3D.pth \
  --data_path /path/to/input_dir/ \
  --save_path /path/to/output_dir/
```

Running via Docker:

```bash
docker pull ghcr.io/intelligentimaging/fetal-bet2:latest

docker run --rm \
    -v {HOST_DATA_DIR}:/data \
    ghcr.io/intelligentimaging/fetal-bet2:latest \
    --data_path /data/{INPUT_DIR} \
    --save_path /data/{OUTPUT_DIR} \
    --dim {DIM}
```

- `{HOST_DATA_DIR}` — host directory mounted into the container as `/data`; it should
  contain `{INPUT_DIR}`, and `{OUTPUT_DIR}` will be written inside it.
- `{DIM}` — `2` for T2w stacks, `3` for dMRI/fMRI.

## Fine-tuning on New Data

Training data (not included in this repository — MRI data is not distributed here)
should be organized under a `dataset/` directory as follows; this is the layout the
released weights were trained on (~1,600 3D volumes / ~55,000 2D slices):

```
dataset/
├── volume_all/           # 3D training volumes (.nii.gz) — one file per case
├── volume_all_mask/       # matching 3D brain masks — same filenames as volume_all/
├── slice_all/             # 2D training slices (.nii.gz), extracted from volume_all/
├── slice_all_mask/         # matching 2D masks — same filenames as slice_all/
├── train_data.csv          # image,label pairs for 3D training (volume_all / volume_all_mask)
└── train_data_slice.csv    # image,label pairs for 2D training (slice_all / slice_all_mask)
```

Each CSV has a header row `image,label` followed by one `<image_path>,<label_path>`
row per case, e.g.:

```
image,label
/abs/path/to/volume_all/T2W_vol0001.nii.gz,/abs/path/to/volume_all_mask/T2W_vol0001.nii.gz
```

Masks are single-channel with the same 2 classes as the released model
(background = 0, brain = 1).

To fine-tune on your own data:

1. Add your new image/mask pairs as `.nii.gz` files into `volume_all`/`volume_all_mask`
   (3D) and/or `slice_all`/`slice_all_mask` (2D).
2. Append the new pairs to `train_data.csv`/`train_data_slice.csv` — or point
   `train_data_paths` in `config_imagine.yml`/`config_imagine_2D.yml` at a new CSV of
   your own, following the same two-column format.
3. Run training with `--pretrained` pointing at the released checkpoint, so training
   starts from the Fetal-BET2 weights instead of a random initialization:

```bash
cd src
python train.py --cfg config_imagine.yml    --pretrained ../Docker/src/models/AttUNet3D.pth
python train.py --cfg config_imagine_2D.yml --pretrained ../Docker/src/models/AttUNet.pth
```

Note: the CSV `image`/`label` paths must resolve on the machine you train on — update
them (or `train_data_paths` in the config) to match wherever `dataset/` actually lives
in your environment.

## Contact

For questions, issues, or feedback, please contact Qinqin Yang at "qinqin.yang@uci.edu" via email.

## Acknowledgement
This research was supported in part by the National Institute of Biomedical Imaging and Bioengineering, the National Institute of Neurological Disorders and Stroke, and Eunice Kennedy Shriver National Institute of Child Health and Human Development of the National Institutes of Health (NIH) under award numbers R01NS106030, R01EB018988, R01EB031849, R01EB032366, and R01HD109395; and in part by the Office of the Director of the NIH under award number S10OD025111. This research was also partly supported by NVIDIA Corporation and utilized NVIDIA RTX A6000 and RTX A5000 GPUs. The content of this publication is solely the responsibility of the authors and does not necessarily represent the official views of the NIH, NSF, or NVIDIA.
