"""
Run a trained U-Net model on the evaluation set.

Each image of datasets/evaluation is a 525 x 526 window of 30 kernels with the
background set to black. The file name <fm>-<rep>a.png gives the true number of
rye kernels (fm) and the replicate (rep), so the true FM ratio is fm / 30.
The model output is resized back to the window size, and its FM ratio is the
green (FM) pixels over the kernel pixels.

    python predict.py                                         the model of the paper (models/unet_best_model.pth)
    python predict.py --model outputs/checkpoints/best_model.pth   a model trained with train.py

Output (outputs/evaluation/): output images, results.csv, summary.txt
"""
import argparse
import csv
import os
import re
import statistics as st

import cv2

from model import fm_ratio, load_model, predict_image

HERE = os.path.dirname(os.path.abspath(__file__))
EVAL = os.path.join(HERE, "datasets", "evaluation")
OUT = os.path.join(HERE, "outputs", "evaluation")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(HERE, "models", "unet_best_model.pth"))
    args = ap.parse_args()
    model = load_model(args.model)
    os.makedirs(os.path.join(OUT, "output_images"), exist_ok=True)

    rows = []
    for name in sorted(os.listdir(EVAL)):
        m = re.match(r"^(\d+)-(\d+)a\.png$", name)
        if not m:
            continue
        fm, rep = int(m.group(1)), int(m.group(2))
        out = predict_image(model, cv2.imread(os.path.join(EVAL, name)))
        cv2.imwrite(os.path.join(OUT, "output_images", name), out)
        true, r = fm / 30, fm_ratio(out)
        rows.append([name, fm, rep, round(true, 6), round(r, 6), round(100 * (r - true), 2)])
    rows.sort(key=lambda x: (x[1], x[2]))
    with open(os.path.join(OUT, "results.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "fm_kernels", "replicate", "true_fm_ratio", "model_fm_ratio", "error_pp"])
        w.writerows(rows)

    e = [x[5] for x in rows]
    bias, sd = st.mean(e), st.stdev(e)
    avg_err = []                                   # error of the replicate average of each sample
    for fm in sorted({x[1] for x in rows}):
        v = [x for x in rows if x[1] == fm]
        avg_err.append(100 * (st.mean(x[4] for x in v) - fm / 30))
    lines = [f"evaluation set, {len(rows)} images",
             f"  bias {bias:+.2f} pp, 95% error range {bias - 1.96 * sd:+.2f} to {bias + 1.96 * sd:+.2f} pp",
             f"  mean absolute error, one image {st.mean(abs(x) for x in e):.2f} pp",
             f"  mean absolute error, replicate average {st.mean(abs(x) for x in avg_err):.2f} pp "
             f"(largest {max(abs(x) for x in avg_err):.2f} pp)"]
    open(os.path.join(OUT, "summary.txt"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
