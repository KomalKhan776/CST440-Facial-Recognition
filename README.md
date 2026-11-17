# Face Recognition with Transfer Learning (MobileNetV2) + Arduino Nano

A face recognition project that learns to identify **4 people** — Komal, Luke, Noah and Reece — from webcam photos. It also includes a model sized for the **Arduino Nano 33 BLE Sense Rev2**. The current board model (**iter5**, int8) scores **96.2 %** on the held-out test set; see [Board model (iter5)](#board-model-iter5-mobilenetv2-035--quantization-aware-training).
It uses a **MobileNetV2** convolutional neural network pre-trained on ImageNet (transfer learning) with a new classifier on top, built with **TensorFlow / Keras**. Faces are found with OpenCV's **YuNet** face detector.

## For teammates – where to start

| Your job | Start with |
|---|---|
| Retrain / improve the model | `preprocess.py --per_person 40` → `train.py` (big model) or `train_tiny.py` (Arduino model) |
| Retrain the board model | [Board model (iter5)](#board-model-iter5-mobilenetv2-035--quantization-aware-training): `train_mnet.py` → `train_qat.py` |
| Deploy to the Nano 33 BLE Sense Rev2 | `arduino/face_model_data.h` (iter5, int8, includes the Unknown threshold) + the framing rules in [Hand-off](#hand-off-for-quantization-and-deployment) |
| Test a model on the held-out photos | `eval_test.py <model> <test folder> --errors` |
| Demo on a laptop | `models/face_model.keras` → `predict.py` / `webcam.py` |

Setup on any computer: `python -m venv .venv`, activate it, `pip install -r requirements.txt` (details below).

**Photos are not in this repository** — they are in the team's shared Google Drive folder. Download them and put them here, one folder per person, then run `python preprocess.py --per_person 40`:
```
dataset/raw/Komal/*.jpg
dataset/raw/Luke/*.jpg
dataset/raw/Noah/*.jpg
dataset/raw/Reece/*.jpg
```

## How it works

```
raw photos ──► detect face (YuNet) ──► crop square ──► resize 224×224 ──► MobileNetV2 ──► Komal / Luke / Noah / Reece
 (3 sizes)        preprocess.py                                            train.py      predict.py / webcam.py
```

1. **Preprocessing** – the original photos come in three resolutions (1280×720, 640×480, 320×240). `preprocess.py` detects the face in every photo, crops it to a square and resizes it, so **every image becomes 224×224 pixels**.
2. **Training** – `train.py` trains in two phases:
   - *Phase 1:* MobileNetV2 is frozen and only the new classification layer is trained.
   - *Phase 2 (fine-tuning):* the top 30 layers of MobileNetV2 are unfrozen and trained with a very small learning rate.
   - Data augmentation (flip, rotation, zoom, brightness, contrast) helps the model generalise from a small dataset.
3. **Recognition** – `predict.py` recognises a face in a photo; `webcam.py` does it live from the webcam. If the confidence is below 60 % the face is labelled **Unknown**.

## Dataset

| Person | Photos taken | Used for training |
|--------|------:|------:|
| Komal  | 55 | 40 |
| Luke   | 40 | 40 |
| Noah   | 40 | 40 |
| Reece  | 40 | 40 |
| **Total** | **175** | **160** |

To keep the classes balanced, `python preprocess.py --per_person 40` uses 40 photos per person (spread evenly over each person's photo session). 80 % of the images are used for training and 20 % for validation.

## Project structure

```
face-recognition-project/
├── dataset/                 # NOT on GitHub - photos come from the team Google Drive
│   ├── raw/                 # original photos, one folder per person
│   └── processed/           # 224x224 face crops made by preprocess.py
├── models/
│   ├── face_detection_yunet_2023mar.onnx   # OpenCV face detector
│   ├── face_model.keras     # trained model (created by train.py)
│   ├── class_names.json     # label order (created by train.py)
│   ├── training_curves.png  # accuracy/loss plot (created by train.py)
│   ├── confusion_matrix.png # validation results (created by train.py)
│   ├── tiny/                # first Arduino-sized model (created by train_tiny.py)
│   ├── iter5_float/         # board model before QAT (train_mnet.py)
│   └── iter5/               # board model, int8 (train_qat.py) - this is what the Nano runs
├── face_utils.py            # face detection + cropping helpers
├── preprocess.py            # step 1: crop & resize all images to 224x224
├── train.py                 # step 2: train the model
├── predict.py               # step 3: recognise a face in one photo
├── webcam.py                # step 4: live recognition from webcam
├── train_tiny.py            # tiny 96x96 grayscale model for the Arduino Nano
├── test_tiny.py             # checks the tiny model uses faces, not backgrounds
├── train_mnet.py            # board model: MobileNetV2-0.35, 96x96 gray (float)
├── train_qat.py             # quantization-aware training -> int8 .tflite + Arduino header
├── quantize.py              # post-training int8 quantization + C header writer
├── eval_test.py             # accuracy on a held-out test folder (board's Unknown rule)
├── make_noperson.py         # builds the NoPerson (empty frame) class from photo backgrounds
├── make_unknown_lfw.py      # adds LFW strangers to the Unknown class
├── capture_empty.py         # webcam tool for extra NoPerson photos
├── requirements.txt
└── README.md
```

## Setup and training in VS Code (Windows)

### 1. Install the tools (once)
- **Python 3.10, 3.11 or 3.12** from [python.org](https://www.python.org/downloads/) — tick **"Add python.exe to PATH"** during installation. (TensorFlow does not always support the newest Python release yet.)
- **VS Code** from [code.visualstudio.com](https://code.visualstudio.com/), then install the **Python** extension (Extensions tab → search "Python" → Install).
- **Git** from [git-scm.com](https://git-scm.com/download/win) (needed for GitHub).

### 2. Open the project
VS Code → **File → Open Folder…** → choose the `face-recognition-project` folder.
Open a terminal with **Terminal → New Terminal** (or <kbd>Ctrl</kbd>+<kbd>`</kbd>).

### 3. Create a virtual environment and install packages
```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```
> If PowerShell says *"running scripts is disabled"*, run
> `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, then activate again.

You should now see `(.venv)` at the start of the terminal line. Press <kbd>Ctrl</kbd>+<kbd>Shift</kbd>+<kbd>P</kbd> → **Python: Select Interpreter** → choose the one inside `.venv`.

### 4. Preprocess the images (make them all 224×224)
Download the photos from the team Google Drive into `dataset/raw/<PersonName>/`, then:
```powershell
python preprocess.py --per_person 40
```

### 5. Train the model
```powershell
python train.py
```
Training takes a few minutes on a normal laptop CPU. At the end it prints the validation accuracy and saves the model and plots in `models/`.
Useful options: `python train.py --epochs 20 --fine_tune_epochs 10 --batch 16`

### 6. Test it
```powershell
python predict.py dataset/processed/Komal/Komal_000.jpg
python webcam.py        # press q to quit
```

### Adding a new person
Create `dataset/raw/<NewName>/` with ~40 photos of their face (different angles and lighting), then run `preprocess.py` and `train.py` again.

## Results

MobileNetV2 reached **100 % validation accuracy** (35/35 images) after about 3 minutes of training on a laptop CPU.

After training, see:
- `models/training_curves.png` – training vs validation accuracy and loss
- `models/confusion_matrix.png` – which people get confused with each other

## Board model (iter5): MobileNetV2-0.35 + quantization-aware training

The model on the Nano is MobileNetV2 at width 0.35 (96×96 grayscale input, ImageNet weights), trained
with `train_mnet.py`, then **quantization-aware trained** (`train_qat.py`) and converted to full int8.
Classes: `Komal, Luke, NoPerson, Noah, Reece, Unknown`.

**Result on the held-out test set** (`dataset/v4/test`: 131 photos from sessions not used for training,
including a stranger and an empty room that are not in the training data), int8 model as it runs on the board.
Board rule: if the top probability is below the threshold the answer is *Unknown* (a top class of NoPerson is kept).

| Threshold | iter5 overall | Komal | Luke | Noah | Reece | Unknown | NoPerson | 3 seeds: mean / worst |
|---|---|---|---|---|---|---|---|---|
| 0.4 | 93.1 % | 100 | 100 | 90 | 80 | 90 | 100 | 89.6 % / 87.0 % |
| 0.45 | 94.7 % | 100 | 100 | 90 | 80 | 95 | 100 | 92.6 % / 90.8 % |
| 0.5 | 96.2 % | 100 | 100 | 90 | 80 | 100 | 100 | 93.9 % / 91.6 % |
| **0.55 (used)** | **96.2 %** | 100 | 100 | 90 | 80 | 100 | 100 | **94.4 % / 92.4 %** |
| 0.6 | 94.7 % | 100 | 95 | 90 | 70 | 100 | 100 | 92.9 % / 89.3 % |
| 0.65 | 93.1 % | 100 | 90 | 90 | 60 | 100 | 100 | 90.3 % / 85.5 % |
| 0.7 | 89.3 % | 100 | 90 | 83 | 30 | 100 | 100 | 87.5 % / 83.2 % |

The threshold (0.55) was chosen on the **mean over 3 training seeds**, not on the best run. iter5 is the
best of those 3 seeds (seed 2), so expect about the mean column (~94 %) from a retrain.
Remaining misses: 3 of Noah's darkest test photos fall under the threshold (→ Unknown) and 2 of Reece's
photos from a different session (one → Noah, one → Unknown).

**From 79 % to 96 % (int8, same test set):**

| Step | int8 test accuracy |
|---|---|
| iter4: deep fine-tuning (`--unfreeze 100 --ls 0.1`) + normal int8 conversion | 68–79 % (float 89–93 %) |
| + quantization-aware training (QAT) | 90–92 % |
| + NoPerson class (100 background crops, `make_noperson.py`) | 88–96 %, mean 92 % |
| + 60 new Reece photos (dim, 2 rooms, head tilts) | Reece 65 → 80 %, but strangers drop to ~50 % |
| + 250 more LFW strangers (`make_unknown_lfw.py`) + longer fine-tuning (`--ft_patience 15`) | **92–96 %, mean 94 %** |

Why each step:
- **QAT.** Normal post-training int8 lost 10–25 points: every ReLU6 gets a fixed 0–6 range and the rounding
  errors add up over ~65 layers (Noah collapsed to Luke). QAT trains with simulated int8 rounding so the weights
  adapt to it. `train_qat.py` rebuilds the network flat (tfmot cannot quantize nested models), freezes BatchNorm
  and sets every activation range from real data before training; without these, QAT collapses to one class.
- **Strangers.** The Unknown class had only 2 board-camera strangers (both men). After Reece's new photos the
  model used "dark hair framing the face" as a shortcut for Reece, so the test stranger (a woman with dark hair)
  became Reece. 250 more LFW faces, 150 of them chosen for dark hair around the face, removed the shortcut.
- **Fine-tuning length.** With early-stopping patience 8 on a ~150-image validation set, fine-tuning stopped
  after ~9 of 40 epochs. `--ft_patience 15` lets it run (16–40 epochs).
- All 40 Unknown test photos are one person, so the Unknown score moves in big jumps. Testing more strangers
  on the board is the best next check.

**Fits on the Nano:** int8 model 618 KB, largest activation 108 KB. A test sketch with the Mbed core +
`arduino/face_model_data.h` compiles to **718 KB of 983 KB flash (73 %)** with 217 KB of RAM free, so about
265 KB of flash remains for TensorFlow Lite Micro and the camera code.

**Reproduce** (photos from the team Google Drive; `dataset/v6/train_np` = iter2 photos + Reece session 2 +
NoPerson + LFW extras; `dataset/v4/test` = held-out test photos):
```powershell
pip install -r requirements.txt
python make_noperson.py        # NoPerson crops -> dataset/v4/train|test/NoPerson
python make_unknown_lfw.py     # LFW strangers -> dataset/lfw_extra/Unknown (download command in the script)
python train_mnet.py --data dataset/v6/train_np --out models/iter5_float --balanced --unfreeze 100 --ft_lr 1e-4 --ls 0.1 --fine_tune_epochs 40 --ft_patience 15 --seed 2
python train_qat.py --float_dir models/iter5_float --data dataset/v6/train_np --out models/iter5 --balanced --test_dir dataset/v4/test --header arduino/face_model_data.h
python eval_test.py models/iter5/tiny_face_model_int8.tflite dataset/v4/test --errors
```
About 5 min for the float model and 15 min for QAT on a laptop CPU; results vary by a few points between seeds.

## Tiny model for the Arduino Nano 33 BLE Sense Rev2

*(First board model, kept for comparison. The deployed model is now [iter5](#board-model-iter5-mobilenetv2-035--quantization-aware-training).)*

The MobileNetV2 model is ~9 MB, but the Nano 33 BLE Sense Rev2 (nRF52840) has only **1 MB flash and 256 KB RAM**. `train_tiny.py` trains a small CNN made for the board:

| | MobileNetV2 (`train.py`) | Tiny CNN (`train_tiny.py`) |
|---|---|---|
| Input | 224×224 RGB | **96×96 grayscale** |
| Parameters | ~2.3 million | **20,772** |
| Size | ~9 MB (float) | 81 KB float · **~47 KB int8** |
| Validation accuracy | 100 % | 97 % (34/35) |
| Fits on Nano | ✗ | ✓ |

```powershell
python train_tiny.py     # ~1 minute, saves to models/tiny/
python test_tiny.py      # checks the model recognises faces, not backgrounds
```

### Does it recognise faces, or the room?
Every person was photographed in a different room, so a model can "cheat" by learning the background. The first version of the tiny model did exactly that: when a face was pasted onto another person's room, accuracy fell to 35 %. Two fixes are built into `train_tiny.py`:
1. **Tight crop** – only the central 75 % of each face crop is used, so the image is mostly face.
2. **Background augmentation** – during training, the background around the face is randomly replaced with a grey level or with another photo's background.

Result of `test_tiny.py`:

| Test | Accuracy |
|---|---|
| Original images | 98.9 % |
| Background → grey | 95.4 % |
| Background → black | 88.6 % |
| Background → white | 77.1 % |
| Face pasted on another person's room | 92.0 % |

(Chance level is 25 %.)

### Hand-off for quantization and deployment
Files in `models/tiny/`:
- `tiny_face_model.keras` and `saved_model/` – the float model, ready to be converted/quantized to TensorFlow Lite (int8).
- `class_names.json` – output order: `Komal, Luke, Noah, Reece`.
- `model_info.json` – input/output specification.

Model specification:
- **Input:** `[1, 96, 96, 1]` grayscale, pixel values **0–255** (the model divides by 255 itself).
- **Output:** `[1, 4]` softmax probabilities in the order above.
- **Ops used:** Conv2D, DepthwiseConv2D, BatchNorm (folded into conv on conversion), ReLU, Mean (global average pooling), FullyConnected, Softmax – all supported by TensorFlow Lite Micro.
- **Memory (int8, measured in a test conversion):** ~47 KB of flash for the model; largest activation tensor 36 KB, so the tensor arena should fit in well under 100 KB of RAM.
- **Framing on the board:** there is no face detector on the Nano, so the camera image must look like the training images. Take a **square centre crop** of the camera frame (for example 120×120 from the OV7675's 160×120 QQVGA frame), convert it to grayscale and resize it to 96×96. The person's face should **fill almost the whole square** (about 30–50 cm from the camera).
- **Representative data** for int8 calibration: the images in `dataset/processed/` (made by `preprocess.py --per_person 40` from the Google Drive photos), put through the same centre-75 % crop → grayscale → 96×96 steps (see `test_tiny.py` → `load()`).

### Quantization (int8 TensorFlow Lite)
```powershell
python quantize.py                       # proper int8, checks accuracy against 80 %
python quantize.py --target 0.95         # same, against the 95 % target
python quantize.py --calibration naive   # baseline experiment: calibrate on random noise
```
`quantize.py` converts `tiny_face_model.keras` to a **full-integer int8** TFLite model, calibrating with the *training* images only (same split and preprocessing as `train_tiny.py`), then compares float vs int8 accuracy on the unseen validation images and on the `test_tiny.py` background tests. It writes:
- `models/tiny/tiny_face_model_int8.tflite` – the model for the board
- `models/tiny/quantization_report.json` – sizes, accuracies, quantization parameters, suggested tensor-arena size
- `models/tiny/int8_confusion_matrix.png`
- `arduino/face_model_data.h` – the model as a C array plus `FACE_MODEL_INPUT_SCALE` / `FACE_MODEL_INPUT_ZERO_POINT` (feed `q = round(pixel / scale) + zero_point`; normally just `pixel - 128`), the class names and `FACE_MODEL_UNKNOWN_THRESHOLD` (`train_qat.py --header` writes the same file for iter5)

`--calibration naive` is a documented baseline (int8 ranges set from random noise instead of faces); its files get a `_naive` suffix so they never replace the deployable model.

## Pushing to GitHub

1. Create a new, empty repository on [github.com](https://github.com/new) (don't add a README there — this project already has one).
2. In the VS Code terminal, inside the project folder:
```powershell
git init
git add .
git commit -m "Face recognition project with MobileNetV2"
git branch -M main
git remote add origin https://github.com/<your-username>/<repo-name>.git
git push -u origin main
```
The first time, Git will open a browser window asking you to sign in to GitHub.

`.gitignore` keeps the virtual environment, temporary files and the `dataset/` folder out of the repository — the photos are shared through the team's Google Drive instead. Both trained models **are** uploaded, so teammates can quantize, deploy or demo straight after cloning.

## Technologies
Python · TensorFlow / Keras · TensorFlow Model Optimization (QAT) · TensorFlow Lite Micro · MobileNetV2 · OpenCV (YuNet face detector) · scikit-learn · Matplotlib
