"""
Add more "Unknown" (stranger) faces from the full LFW dataset (Labeled Faces in the Wild, a public
face-recognition research set: ~13k photos of ~5.7k people).

Why: the Unknown class had 150 LFW faces + 2 board-camera roommates (both men). The model learned
"dark hair framing the face" as a shortcut for Reece/Komal, so a dark-haired stranger was called Reece.
More strangers - weighted toward dark hair around the face - break that shortcut.

What it does:
  * 1 photo per person (more identities > more photos of the same person)
  * skips the people already used in dataset/raw_v2/Unknown (lfw_<name>_*)
  * face-crops exactly like preprocess.py (YuNet + square crop with margin, 224x224)
  * scores "dark hair framing the face" = mean brightness of the strips above and beside the face,
    then takes --dark from the darkest third and the rest at random (seeded)

Usage:
    python -c "from sklearn.datasets import fetch_lfw_people as f; f(data_home='dataset/lfw_full', resize=0.2)"
    python make_unknown_lfw.py --n 250 --dark 150 --out dataset/lfw_extra/Unknown
"""
import argparse
import glob
import os
import random

import cv2
import numpy as np

from face_utils import FaceDetector, face_crop


def framing_darkness(crop):
    """Mean brightness of the top strip and the two side strips of a 224 face crop (low = dark hair)."""
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
    H, W = g.shape
    top = g[: int(0.18 * H), int(0.2 * W): int(0.8 * W)]
    left = g[int(0.15 * H): int(0.65 * H), : int(0.16 * W)]
    right = g[int(0.15 * H): int(0.65 * H), int(0.84 * W):]
    return float(np.mean([top.mean(), left.mean(), right.mean()]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lfw", default="dataset/lfw_full/lfw_home/lfw_funneled")
    ap.add_argument("--used", default="dataset/raw_v2/Unknown", help="folder whose lfw_<name>_* people are skipped")
    ap.add_argument("--out", default="dataset/lfw_extra/Unknown")
    ap.add_argument("--n", type=int, default=250)
    ap.add_argument("--dark", type=int, default=150, help="how many of --n come from the darkest-framing third")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # existing files are lfw_<first>_<last>.jpg (lower case), e.g. lfw_afton_smith.jpg
    used = {os.path.splitext(f)[0][len("lfw_"):].lower() for f in os.listdir(args.used) if f.startswith("lfw_")}
    people = sorted(d for d in os.listdir(args.lfw) if os.path.isdir(os.path.join(args.lfw, d)))
    n_all = len(people)
    people = [p for p in people if p.lower() not in used]
    print(f"skipping {n_all - len(people)} of {len(used)} people already in {args.used}")
    rng = random.Random(args.seed)
    rng.shuffle(people)
    print(f"{len(people)} LFW people not used yet; scanning up to {args.n * 6}")

    det = FaceDetector()
    cands = []
    for p in people[: args.n * 6]:
        f = sorted(glob.glob(os.path.join(args.lfw, p, "*.jpg")))[0]
        crop, found = face_crop(cv2.imread(f), det, 224)
        if found:
            cands.append((framing_darkness(crop), p, crop))
    cands.sort(key=lambda c: c[0])
    third = len(cands) // 3
    dark = cands[:third][: args.dark]
    rest = cands[third:]
    rng.shuffle(rest)
    pick = dark + rest[: args.n - len(dark)]

    os.makedirs(args.out, exist_ok=True)
    for old in glob.glob(os.path.join(args.out, "lfw2_*.jpg")):
        os.remove(old)
    for _, p, crop in pick:
        cv2.imwrite(os.path.join(args.out, f"lfw2_{p.lower()}.jpg"), crop)
    print(f"faces found {len(cands)}, saved {len(pick)} ({len(dark)} dark-framing) to {args.out}")
    print(f"framing brightness: darkest third <= {cands[third][0]:.0f}, all-candidate median {cands[len(cands)//2][0]:.0f}")


if __name__ == "__main__":
    main()
