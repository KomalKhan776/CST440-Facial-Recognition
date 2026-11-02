# Face Recognition with Transfer Learning (MobileNetV2) + Arduino Nano

A face recognition project that learns to identify **4 people** — Komal, Luke, Noah and Reece — from webcam photos. It also includes a tiny model sized for the **Arduino Nano 33 BLE Sense Rev2** (see [Tiny model for the Arduino](#tiny-model-for-the-arduino-nano-33-ble-sense-rev2)).
It uses a **MobileNetV2** convolutional neural network pre-trained on ImageNet (transfer learning) with a new classifier on top, built with **TensorFlow / Keras**. Faces are found with OpenCV's **YuNet** face detector.

## For teammates – where to start

| Your job | Start with |
|---|---|
| Retrain / improve the model | `preprocess.py --per_person 40` → `train.py` (big model) or `train_tiny.py` (Arduino model) |
| Quantize for the Arduino | `models/tiny/tiny_face_model.keras` or `models/tiny/saved_model/` + the spec in [Hand-off for quantization and deployment](#hand-off-for-quantization-and-deployment) |
| Deploy to the Nano 33 BLE Sense Rev2 | the quantized model + the input/framing rules in the same section |
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
│   └── tiny/                # Arduino-sized model (created by train_tiny.py)
├── face_utils.py            # face detection + cropping helpers
├── preprocess.py            # step 1: crop & resize all images to 224x224
├── train.py                 # step 2: train the model
├── predict.py               # step 3: recognise a face in one photo
├── webcam.py                # step 4: live recognition from webcam
├── train_tiny.py            # tiny 96x96 grayscale model for the Arduino Nano
├── test_tiny.py             # checks the tiny model uses faces, not backgrounds
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

## Tiny model for the Arduino Nano 33 BLE Sense Rev2

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
Python · TensorFlow / Keras · MobileNetV2 · OpenCV (YuNet face detector) · scikit-learn · Matplotlib
