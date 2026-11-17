"""
Split the labeled patches, train the U-Net model, and test it.

1. Split datasets/labeled into test, validation and training patches.
     --split random   (default) draws them at random with --seed (24 test, 10 validation)
     --split paper    uses datasets/labeled/paper_split.csv, the split of the paper
2. Augment the training patches: each is used six times (original, horizontal
   flip, vertical flip, rotations by 90, 180 and 270 degrees).
3. Train. Settings of the paper: L1 loss x 100, Adam (learning rate 2e-4, betas
   0.5 and 0.999), batch size 4, up to 100 epochs. The validation loss is computed
   after every epoch. The learning rate is halved after 5 epochs without
   improvement, training stops after 15, and the weights with the lowest
   validation loss are kept.
4. Test the kept weights on the test patches. The FM ratio of the model output
   is compared with the FM ratio of the label image.

    python train.py                        random split, full training (about 12 h on a 4-core CPU)
    python train.py --split paper          the split of the paper
    python train.py --limit 40 --epochs 1  quick test of the set-up

Output (outputs/): split.csv, train_log.csv, checkpoints/best_model.pth and
last_model.pth, test_results.csv, test_summary.txt.
Training itself is not seeded, so each run gives slightly different weights.
"""
import argparse
import csv
import os
import random
import statistics as st
import time

import cv2
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset

from model import UNet, fm_ratio, load_model, predict_image, to_tensor

HERE = os.path.dirname(os.path.abspath(__file__))
LAB = os.path.join(HERE, "datasets", "labeled")
OUT = os.path.join(HERE, "outputs")
BATCH_SIZE, LR, PATIENCE = 4, 0.0002, 15


def augment(img, k):
    if k == 0: return img
    if k == 1: return cv2.flip(img, 1)
    if k == 2: return cv2.flip(img, 0)
    if k == 3: return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if k == 4: return cv2.rotate(img, cv2.ROTATE_180)
    if k == 5: return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)


class PatchDataset(Dataset):
    """input_a and label pairs, each used n_aug times (1 = no augmentation)."""
    def __init__(self, names, n_aug):
        self.items = [(f, k) for f in names for k in range(n_aug)]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        f, k = self.items[i]
        a = augment(cv2.imread(os.path.join(LAB, "input_a", f)), k)
        b = augment(cv2.imread(os.path.join(LAB, "label", f)), k)
        return to_tensor(a, cv2.INTER_LINEAR), to_tensor(b, cv2.INTER_NEAREST)


def make_split(how, seed, n_test, n_val):
    names = sorted(f for f in os.listdir(os.path.join(LAB, "input_a")) if f.endswith(".png"))
    if how == "paper":
        s = {r["image"]: r["split"] for r in csv.DictReader(open(os.path.join(LAB, "paper_split.csv")))}
        pick = lambda k: [n for n in names if s[n] == k]
        return pick("training"), pick("validation"), pick("test")
    random.Random(seed).shuffle(names)
    test, val, train = names[:n_test], names[n_test:n_test + n_val], names[n_test + n_val:]
    return sorted(train), sorted(val), sorted(test)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["random", "paper"], default="random")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-test", type=int, default=24)
    ap.add_argument("--n-val", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--limit", type=int, default=None, help="train on only the first N training patches")
    args = ap.parse_args()
    os.makedirs(os.path.join(OUT, "checkpoints"), exist_ok=True)

    # 1. split
    train, val, test = make_split(args.split, args.seed, args.n_test, args.n_val)
    if args.limit:
        train = train[:args.limit]
    with open(os.path.join(OUT, "split.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["image", "split"])
        w.writerows(sorted([(n, "training") for n in train] + [(n, "validation") for n in val] +
                           [(n, "test") for n in test]))
    print(f"split ({args.split}): training {len(train)}, validation {len(val)}, test {len(test)}")

    # 2. augment and 3. train
    train_loader = DataLoader(PatchDataset(train, 6), BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(PatchDataset(val, 1), BATCH_SIZE, shuffle=False, num_workers=0)
    model = UNet()
    optimizer = Adam(model.parameters(), lr=LR, betas=(0.5, 0.999))
    scheduler = ReduceLROnPlateau(optimizer, "min", patience=5, factor=0.5)
    criterion = nn.L1Loss()
    print(f"training pairs after augmentation {len(train_loader.dataset)}, "
          f"parameters {sum(p.numel() for p in model.parameters()):,}")
    log_path = os.path.join(OUT, "train_log.csv")
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "val_loss", "time_s"])
    best_val, wait = float("inf"), 0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train(); tr = 0.0
        for a, b in train_loader:
            loss = criterion(model(a), b) * 100
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            tr += loss.item()
        tr /= len(train_loader)
        model.eval(); va = 0.0
        with torch.no_grad():
            for a, b in val_loader:
                va += (criterion(model(a), b) * 100).item()
        va /= len(val_loader)
        scheduler.step(va)
        t = time.time() - t0
        print(f"epoch {epoch:>3}  training loss {tr:.4f}  validation loss {va:.4f}  {t:.0f} s")
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, round(tr, 4), round(va, 4), round(t, 1)])
        torch.save(model.state_dict(), os.path.join(OUT, "checkpoints", "last_model.pth"))
        if va < best_val:
            best_val, wait = va, 0
            torch.save(model.state_dict(), os.path.join(OUT, "checkpoints", "best_model.pth"))
            print("  lowest validation loss so far, weights saved")
        else:
            wait += 1
            if wait >= PATIENCE:
                print(f"stopped, no improvement for {PATIENCE} epochs")
                break

    # 4. test
    best = load_model(os.path.join(OUT, "checkpoints", "best_model.pth"))
    rows = []
    for n in test:
        out = predict_image(best, cv2.imread(os.path.join(LAB, "input_a", n)))
        lab = fm_ratio(cv2.imread(os.path.join(LAB, "label", n)))
        r = fm_ratio(out)
        rows.append([n, round(lab, 6), round(r, 6), round(100 * (r - lab), 2)])
    with open(os.path.join(OUT, "test_results.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["image", "label_fm_ratio", "model_fm_ratio", "error_pp"]); w.writerows(rows)
    msg = (f"lowest validation loss {best_val:.4f}\n"
           f"test set, {len(rows)} patches: mean absolute error of the FM ratio "
           f"{st.mean(abs(x[3]) for x in rows):.2f} percentage points")
    open(os.path.join(OUT, "test_summary.txt"), "w").write(msg + "\n")
    print(msg)


if __name__ == "__main__":
    main()
