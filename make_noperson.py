"""
Build a "NoPerson" (empty frame) class from the background of existing photos.

The board has no face detector, so an empty frame must still map to *something*. This script cuts
face-free square patches out of the raw photos (dataset/raw_v2/<Name>/*.jpg):
  1. YuNet finds every face (low threshold, so faint faces are avoided too).
  2. An exclusion zone covers the face box plus margin and the body below it (shoulders/torso),
     so patches show room only, not a shirt.
  3. Random squares (20-60 % of the short image side) that do not touch the zone are cut out,
     re-checked with YuNet (rejected if any face is found), resized to 224x224 and saved like
     preprocess.py output (the trainer then applies the same gray -> central 75 % -> 96x96 steps).

A whole room is held out for the test set (default: Noah's room) so the test measures new rooms.
A few synthetic "covered lens / dark room / blank wall" frames are added to training only.

Usage:
    python make_noperson.py                      # -> dataset/v4/train/NoPerson, dataset/v4/test/NoPerson
    python make_noperson.py --per_room 25 --test_room noah
"""
import argparse
import os
import random

import cv2
import numpy as np

from face_utils import FaceDetector

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def room_of(fname):
    """Room/session id from the raw file name, e.g. luke_s1_apt_001.jpg -> apt, noah_s0_... -> noah."""
    f = fname.lower()
    if "_apt_" in f:
        return "apt"                      # Luke s1 + roommates share the apartment
    if f.startswith("lfw_"):
        return None                       # LFW backgrounds are tiny 250x250 portraits: skip
    return f.split("_")[0] + "_" + f.split("_")[1]   # komal_s0, luke_s0, noah_s0, reece_s0


def exclusion_mask(shape, faces):
    H, W = shape[:2]
    m = np.zeros((H, W), bool)
    for x, y, w, h, _ in faces:
        x1, x2 = int(x - 0.5 * w), int(x + 1.5 * w)        # face + margin, wide enough for hair/ears
        y1 = int(y - 0.5 * h)
        m[max(0, y1):H, max(0, x1):min(W, x2)] = True        # everything below too (neck, torso)
        bx1, bx2 = int(x - 1.0 * w), int(x + 2.0 * w)        # shoulders are wider than the face
        m[max(0, int(y + 1.2 * h)):H, max(0, bx1):min(W, bx2)] = True
    return m


def crops_from_image(img, detector, rng, n_tries=300):
    H, W = img.shape[:2]
    faces = detector.detect(img)
    mask = exclusion_mask(img.shape, faces)
    out = []
    for _ in range(n_tries):
        s = int(rng.uniform(0.2, 0.6) * min(H, W))
        x, y = rng.randint(0, W - s), rng.randint(0, H - s)
        if mask[y:y + s, x:x + s].any():
            continue
        patch = img[y:y + s, x:x + s]
        if detector.detect(patch):                          # paranoid re-check on the patch itself
            continue
        if patch.std() < 4:                                  # skip completely flat blown-out patches
            continue
        out.append(cv2.resize(patch, (224, 224), interpolation=cv2.INTER_AREA))
    return out


def synthetic(rng, n):
    """Covered lens / dark room / plain wall frames (training only)."""
    out = []
    for i in range(n):
        base = [5, 20, 60, 120, 180, 230][i % 6] + rng.uniform(-10, 10)
        gy, gx = np.mgrid[:224, :224] / 224.0
        grad = (gx * rng.uniform(-40, 40) + gy * rng.uniform(-40, 40))
        im = np.clip(base + grad + np.random.default_rng(i).normal(0, rng.uniform(1, 6), (224, 224)), 0, 255)
        out.append(cv2.cvtColor(im.astype(np.uint8), cv2.COLOR_GRAY2BGR))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="dataset/raw_v2")
    ap.add_argument("--train_out", default="dataset/v4/train/NoPerson")
    ap.add_argument("--test_out", default="dataset/v4/test/NoPerson")
    ap.add_argument("--test_room", default="noah_s0")
    ap.add_argument("--per_room", type=int, default=23, help="training crops per room")
    ap.add_argument("--test_n", type=int, default=20)
    ap.add_argument("--synthetic", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    det = FaceDetector(score_threshold=0.5)
    by_room = {}
    for person in sorted(os.listdir(args.raw)):
        d = os.path.join(args.raw, person)
        for f in sorted(os.listdir(d)):
            r = room_of(f) if f.lower().endswith(IMG_EXTS) else None
            if r:
                by_room.setdefault(r, []).append(os.path.join(d, f))

    for out in (args.train_out, args.test_out):
        os.makedirs(out, exist_ok=True)
        for old in os.listdir(out):
            os.remove(os.path.join(out, old))

    for room, files in sorted(by_room.items()):
        is_test = room == args.test_room
        want = args.test_n if is_test else args.per_room
        rng.shuffle(files)
        got = []
        # at most 3 crops per photo so the patches spread over the whole session
        for f in files:
            img = cv2.imread(f)
            if img is None:
                continue
            got += crops_from_image(img, det, rng)[:3]
            if len(got) >= want:
                break
        got = got[:want]
        dst = args.test_out if is_test else args.train_out
        for i, c in enumerate(got):
            cv2.imwrite(os.path.join(dst, f"noperson_{room}_{i:03d}.jpg"), c)
        print(f"{room:<10} {'TEST ' if is_test else 'train'} {len(got)} crops from {len(files)} photos")

    for i, im in enumerate(synthetic(rng, args.synthetic)):
        cv2.imwrite(os.path.join(args.train_out, f"noperson_synth_{i:03d}.jpg"), im)
    print(f"synthetic  train {args.synthetic}")
    print(f"\ntrain: {len(os.listdir(args.train_out))}  test: {len(os.listdir(args.test_out))}")


if __name__ == "__main__":
    main()
