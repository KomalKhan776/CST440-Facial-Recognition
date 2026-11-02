"""
Check that the tiny (Arduino) model really recognises FACES, not the room behind them.

Each person was photographed in a different place, so a model can "cheat" by
learning the background. This script keeps only the face (an ellipse in the
middle of the image) and replaces the background with black, grey, white, or
the background of a *different* person's photo. A model that uses the face
keeps a high accuracy; one that uses the background collapses to ~25 %.

Usage:
    python test_tiny.py
"""
import argparse
import glob
import json
import os

import numpy as np
import tensorflow as tf
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default="models/tiny")
    ap.add_argument("--data", default="dataset/processed")
    ap.add_argument("--size", type=int, default=96)
    ap.add_argument("--face_fraction", type=float, default=0.75)
    args = ap.parse_args()

    names = json.load(open(os.path.join(args.model_dir, "class_names.json")))
    model = tf.keras.models.load_model(os.path.join(args.model_dir, "tiny_face_model.keras"))
    files = sorted(glob.glob(os.path.join(args.data, "*", "*.jpg")))
    y = np.array([names.index(os.path.basename(os.path.dirname(f))) for f in files])

    def load(f):
        im = Image.open(f).convert("L")
        w, h = im.size
        a = int(w * (1 - args.face_fraction) / 2)
        im = im.crop((a, a, w - a, h - a)).resize((args.size, args.size), Image.BOX)
        return np.array(im, dtype=np.float32)

    X = np.stack([load(f) for f in files])
    s = args.size
    yy, xx = np.mgrid[:s, :s]
    face = ((xx - s / 2) / (0.40 * s)) ** 2 + ((yy - 0.52 * s) / (0.48 * s)) ** 2 <= 1

    def acc(imgs):
        return np.mean(model.predict(imgs[..., None], verbose=0).argmax(1) == y)

    print(f"{'test':<34}accuracy")
    print(f"{'original images':<34}{acc(X):.1%}")
    for v, name in ((0, "black"), (128, "grey"), (255, "white")):
        B = X.copy(); B[:, ~face] = v
        print(f"{'background -> ' + name:<34}{acc(B):.1%}")
    rng = np.random.default_rng(0)
    S = []
    for i in range(len(X)):
        j = rng.choice(np.where(y != y[i])[0])
        b = X[j].copy(); b[face] = X[i][face]; S.append(b)
    print(f"{'face pasted on another room':<34}{acc(np.stack(S)):.1%}")
    print("\nChance level is 25 % (4 people). Values close to the original mean the model uses the face.")


if __name__ == "__main__":
    main()
