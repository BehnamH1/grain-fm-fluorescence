# Data sets

Wheat is hard red winter wheat, the foreign material (FM) is rye. All images are
PNG, with the background set to black, ready for the model.

## labeled/

431 labeled patches of about 525 x 526 pixels, cut from photographs of
single-layer samples. The rye kernels were stained with an ink that is invisible
under ordinary light and glows blue under UV light, and each sample was
photographed under both lights without being moved. `train.py` splits these
patches into training, validation and test patches.

| Folder | Content |
|---|---|
| `input_a/` | Ordinary-light patch, the input of the model. |
| `uv_b/` | UV patch of the same sample. |
| `label/` | Label image made from the UV patch, the training target. Green (0, 255, 0) = rye, gray (128, 128, 128) = wheat, black = background. |
| `paper_split.csv` | The split used in the paper: 397 training, 10 validation and 24 test patches (one test patch per number of kernels per patch). |

The three folders hold the same file names. Names start with `ch1_` (batch A,
photographs of 3 x 4 cells) or `ch3_` (batch B, 3 x 5 cells).
`<n>_patch_<row>_<col>` gives the photograph and the cell, and `_k<k>` (batch B
only) the number of kernels in the cell, from 1 to 50.

## evaluation/

42 windows of 525 x 526 pixels, each holding 30 kernels, taken after training.
No ink was used, so there is no UV or label image. The file name
`<fm>-<rep>a.png` gives the truth. `fm` is the number of rye kernels among the
30 (0 to 10), so the true FM ratio is fm / 30, and `rep` is the replicate, a
re-arrangement of the same kernels (three replicates for fm = 0 and 3, four for
the others).
