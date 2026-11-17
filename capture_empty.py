"""
Take "NoPerson" (empty frame) photos with the laptop webcam for the NoPerson class.

The board has no face detector: it always classifies the central square of the camera frame.
So an empty-frame photo here is the CENTRAL SQUARE of the webcam frame, resized to 224x224
(the same size preprocess.py produces). The full frame is kept too, in case we want to re-crop.

Saves
    dataset/v4/noperson_new/<location>/noperson_<location>_###.jpg       224x224 center square (train on this)
    dataset/v4/noperson_new/<location>/raw/noperson_<location>_###.jpg   full webcam frame

Photos are only saved when NO face is detected, so you can't accidentally save yourself.
Make one --location per room/spot so one room can be held out as a test set.

Keys (in the camera window):
    SPACE  take one photo
    a      start / stop auto mode (one photo every --interval seconds) - slowly pan the laptop
    q      quit

What to capture (aim for 60-100 total, 3+ locations):
    - plain walls, doors, shelves, a chair, a desk - whatever is behind the board in the demo
    - lights on, lights dim, lamp only; daylight and night
    - the camera partly covered by a hand, and fully covered (dark frame)
    - tilt / pan a bit between shots so they are not near-duplicates

Usage:
    python capture_empty.py --location dorm
    python capture_empty.py --location classroom --count 30 --interval 1.0
Then tell Claude (or copy them into dataset/v4/train/NoPerson and retrain).
"""
import argparse
import glob
import os
import re
import time

import cv2

from face_utils import FaceDetector, center_crop

GREEN, RED, WHITE = (0, 200, 0), (0, 0, 255), (255, 255, 255)


def next_index(folder, prefix):
    nums = [int(m.group(1)) for f in glob.glob(os.path.join(folder, prefix + "_*.jpg"))
            if (m := re.search(r"_(\d+)\.jpg$", f))]
    return max(nums, default=0) + 1


def text(img, s, y, color=WHITE, scale=0.7):
    cv2.putText(img, s, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4)
    cv2.putText(img, s, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--location", required=True, help="short room tag, e.g. dorm, apt, classroom, library")
    ap.add_argument("--count", type=int, default=40, help="stop after this many new photos")
    ap.add_argument("--interval", type=float, default=1.0, help="seconds between auto photos")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--out", default="dataset/v4/noperson_new")
    args = ap.parse_args()

    loc = args.location.lower().replace(" ", "")
    folder = os.path.join(args.out, loc)
    raw_folder = os.path.join(folder, "raw")
    os.makedirs(raw_folder, exist_ok=True)
    prefix = f"noperson_{loc}"
    idx = next_index(folder, prefix)

    detector = FaceDetector(score_threshold=0.5)    # stricter than capture.py: reject faint faces too
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit("Could not open the webcam (is another program using it?).")

    saved, auto, last, flash_until = 0, False, 0.0, 0.0
    while saved < args.count:
        ok, frame = cap.read()
        if not ok:
            break
        H, W = frame.shape[:2]
        faces = detector.detect(frame)
        good = len(faces) == 0
        status = "empty frame - OK" if good else "a face is visible - step out of view"

        key = cv2.waitKey(1) & 0xFF
        now = time.time()
        take = (key == ord(" ")) or (auto and now - last >= args.interval)
        if key == ord("a"):
            auto, last = not auto, now
        if key == ord("q"):
            break
        if take and good:
            name = f"{prefix}_{idx:03d}.jpg"
            cv2.imwrite(os.path.join(raw_folder, name), frame)
            sq = cv2.resize(center_crop(frame), (224, 224), interpolation=cv2.INTER_AREA)
            cv2.imwrite(os.path.join(folder, name), sq)
            idx += 1; saved += 1; last = now; flash_until = now + 0.15

        # Preview: show the square the board would see
        view = frame.copy()
        s = min(H, W); x0, y0 = (W - s) // 2, (H - s) // 2
        cv2.rectangle(view, (x0, y0), (x0 + s, y0 + s), GREEN if good else RED, 2)
        for (x, y, w, h, _) in faces:
            cv2.rectangle(view, (x, y), (x + w, y + h), RED, 2)
        if now < flash_until:
            cv2.rectangle(view, (0, 0), (W - 1, H - 1), WHITE, 12)
        text(view, f"NoPerson @ {loc}   saved {saved}/{args.count}   auto: {'ON' if auto else 'off'}", 28,
             GREEN if auto else WHITE)
        text(view, status, 58, GREEN if good else RED)
        text(view, "SPACE photo | a auto (pan slowly) | q quit   - green box = what the board sees",
             H - 15, WHITE, 0.5)
        cv2.imshow("Empty-frame capture - press q to quit", view)

    cap.release()
    cv2.destroyAllWindows()
    print(f"Saved {saved} photos to {folder} (prefix {prefix}_)")


if __name__ == "__main__":
    main()
