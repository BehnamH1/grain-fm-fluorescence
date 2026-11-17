# Fluorescence labeling for deep-learning grain inspection

Data and code for the paper

> Hassanizadeh, B., Kenkel, P. Deep-learning machine vision for automated grain
> grading with fluorescence-based labeling: foreign material in wheat.
> *Computers and Electronics in Agriculture* (submitted). [DOI to be added]

Rye kernels (foreign material, FM) in hard red winter wheat were stained with an
ink that is invisible under ordinary light and glows under UV light. Each
training sample was photographed under both lights, and the UV image was turned
into a label image without manual labeling. A U-Net model learns to produce the
label image from the ordinary-light image alone, so at use only a standard camera
is needed. The FM ratio of a sample is the share of kernel pixels the model marks
as FM.

## Contents

```
datasets/labeled/     431 labeled patches (input, UV image, label)
datasets/evaluation/  42 evaluation images, true rye count in the file name
model.py              the U-Net model and the shared steps
train.py              splits the labeled patches, trains, validates and tests the model
predict.py            runs a trained model on the evaluation set
models/               the trained model of the paper (epoch 31)
outputs/              written by the scripts
```

See `datasets/README.md` for the data.

## Setup

Python 3.11, CPU only (no graphics card needed).

```
pip install -r requirements.txt
```

## Train and test a model

```
python train.py
```

1. The 431 labeled patches are split at random (seed 42) into 24 test, 10
   validation and 397 training patches. `--split paper` uses the split of the
   paper instead, and `--seed`, `--n-test` and `--n-val` change the random split.
2. Each training patch is used six times, as it is, flipped horizontally and
   vertically, and rotated by 90, 180 and 270 degrees.
3. The model is trained with the settings of the paper (L1 loss, Adam with
   learning rate 2e-4, batch size 4). After every epoch the loss on the
   validation patches is computed. The learning rate is halved after 5 epochs
   without improvement, training stops after 15, and the weights with the
   lowest validation loss are kept.
4. The kept weights are tested on the test patches, by comparing the FM ratio
   of the model output with that of the label image.

On a four-core CPU an epoch takes about 16 minutes and a full run about 12
hours. Training is not seeded, so each run gives slightly different weights.
`python train.py --limit 8 --epochs 1` is a one-minute test of the set-up.
Results are written to `outputs/` (`split.csv`, `train_log.csv`,
`checkpoints/best_model.pth`, `test_results.csv`).

## Run a model on the evaluation set

```
python predict.py                                            the model of the paper
python predict.py --model outputs/checkpoints/best_model.pth a model from train.py
```

The model of the paper is `models/unet_best_model.pth` (218 MB, stored with Git
LFS, so clone with Git LFS installed or download the file from the GitHub page). The output images,
`results.csv` and `summary.txt` are written to `outputs/evaluation/`. With the
model of the paper this reproduces the results of the paper.

| Model of the paper | |
|---|---|
| Evaluation set, bias | +0.62 percentage points (pp) |
| Evaluation set, 95% error range of one image | −3.46 to +4.70 pp |
| Evaluation set, mean absolute error of one image | 1.76 pp |
| Evaluation set, mean absolute error of the replicate average | 1.24 pp |
| Test set (paper split), mean absolute error | 7.79 pp |

## How the data sets were made

The photographs were cropped to the sample area, the background was removed by
a threshold on the red channel, and the photographs were cut into patches along
the grid of the sample layout. Touching kernels were separated by a second
network, and each kernel was labeled rye when at least 5% of its pixels glowed
blue in the UV image (hue 87 to 130 on the 0 to 179 scale of OpenCV). The
details are in the appendix of the paper.

## Use of AI

The arrangement of this repository (the folder layout, the selection and
consolidation of the scripts into `model.py`, `train.py` and `predict.py`, and
the documentation) was done with the help of an AI assistant (Anthropic Claude).
The author reviewed the result. The model, the data and the training settings
are those of the study, and `predict.py` reproduces the published results.

## License

Code: MIT (see `LICENSE`). Data and model: CC BY 4.0.

## Citation

Please cite the paper above.
