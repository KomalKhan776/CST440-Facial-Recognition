"""
Step 1 - Preprocess the dataset (make all pictures the same resolution).

Reads every image in  dataset/raw/<PersonName>/*.jpg
  1. detects the face (OpenCV YuNet detector)
  2. crops the face (with a small margin) to a square
  3. resizes it to SIZE x SIZE (default 224x224)
  4. saves it to      dataset/processed/<PersonName>/

If no face is found the image is skipped (use --keep_no_face to center-crop
it instead), so the training data only contains faces.

Usage:
    python preprocess.py
    python preprocess.py --per_person 40     (same number of images for everyone)
    python preprocess.py --raw dataset/raw --out dataset/processed --size 224
"""
import argparse
import os

import cv2

from face_utils import FaceDetector, face_crop

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="dataset/raw")
    ap.add_argument("--out", default="dataset/processed")
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--per_person", type=int, default=None,
                    help="use at most this many images per person, e.g. 40 to balance the classes")
    ap.add_argument("--keep_no_face", action="store_true",
                    help="center-crop images where no face is found instead of skipping them")
    args = ap.parse_args()

    detector = FaceDetector()
    total, skipped = 0, []

    for person in sorted(os.listdir(args.raw)):
        src_dir = os.path.join(args.raw, person)
        if not os.path.isdir(src_dir):
            continue
        dst_dir = os.path.join(args.out, person)
        os.makedirs(dst_dir, exist_ok=True)
        # remove results of an earlier run so old images don't stay behind
        for old in os.listdir(dst_dir):
            if old.lower().endswith(IMG_EXTS):
                os.remove(os.path.join(dst_dir, old))

        files = sorted(f for f in os.listdir(src_dir) if f.lower().endswith(IMG_EXTS))
        n = len(files)
        if args.per_person and n > args.per_person:
            # pick images evenly spread over the whole photo session (keeps pose variety)
            k = args.per_person
            files = [files[round(j * (n - 1) / (k - 1))] for j in range(k)] if k > 1 else files[:1]

        count = 0
        for i, fname in enumerate(files):
            img = cv2.imread(os.path.join(src_dir, fname))
            if img is None:
                print(f"  ! could not read {fname}")
                continue

            crop, found = face_crop(img, detector, args.size)
            if not found:
                skipped.append(os.path.join(person, fname))
                if not args.keep_no_face:
                    continue
            cv2.imwrite(os.path.join(dst_dir, f"{person}_{i:03d}.jpg"), crop)
            count += 1

        total += count
        print(f"{person:<12} {count} images")

    print(f"\nDone: {total} images saved to '{args.out}' at {args.size}x{args.size}.")
    if skipped:
        action = "center-cropped" if args.keep_no_face else "skipped"
        print(f"No face found in {len(skipped)} image(s) ({action}):")
        for s in skipped:
            print("   ", s)


if __name__ == "__main__":
    main()
