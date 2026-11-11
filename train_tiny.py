"""
Tiny model for the Arduino Nano 33 BLE Sense Rev2 (nRF52840: 1 MB flash, 256 KB RAM).

The big model (train.py, MobileNetV2 224x224 RGB) is ~9 MB and cannot run on the
board. This script trains a small CNN that does:

    input : 96 x 96 x 1  grayscale, pixel values 0-255 (float32)
    output: 4 softmax probabilities in the order of models/tiny/class_names.json

It only uses ops supported by TensorFlow Lite Micro (Conv2D, DepthwiseConv2D,
BatchNorm (folded at conversion), ReLU, MaxPool, AveragePool, Reshape,
FullyConnected, Softmax), so it can be quantized to int8 and deployed.

Outputs (models/tiny/):
    tiny_face_model.keras      float model - hand this to the quantization step
    saved_model/               same model in SavedModel format
    class_names.json           label order
    model_info.json            input/output spec, parameter count, accuracy
    tiny_training_curves.png
    tiny_confusion_matrix.png

Usage:
    python train_tiny.py
    python train_tiny.py --epochs 80
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

SEED = 42
L = tf.keras.layers


def sep_block(x, filters, stride):
    x = L.DepthwiseConv2D(3, strides=stride, padding="same", use_bias=False)(x)
    x = L.BatchNormalization(momentum=0.9)(x)
    x = L.ReLU()(x)
    x = L.Conv2D(filters, 1, use_bias=False)(x)
    x = L.BatchNormalization(momentum=0.9)(x)
    return L.ReLU()(x)


def build_tiny_model(num_classes, size):
    inputs = tf.keras.Input(shape=(size, size, 1), name="image")
    x = L.Rescaling(1.0 / 255)(inputs)
    x = L.Conv2D(8, 3, strides=2, padding="same", use_bias=False)(x)   # 48x48
    x = L.BatchNormalization(momentum=0.9)(x)
    x = L.ReLU()(x)
    x = sep_block(x, 16, 1)
    x = sep_block(x, 32, 2)    # 24x24
    x = sep_block(x, 32, 1)
    x = sep_block(x, 64, 2)    # 12x12
    x = sep_block(x, 64, 1)
    x = sep_block(x, 128, 2)   # 6x6
    x = L.GlobalAveragePooling2D()(x)
    x = L.Dropout(0.3)(x)
    outputs = L.Dense(num_classes, activation="softmax", name="probs")(x)
    return tf.keras.Model(inputs, outputs, name="tiny_face_net")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset/processed")
    ap.add_argument("--size", type=int, default=96)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--val_split", type=float, default=0.2)
    ap.add_argument("--face_fraction", type=float, default=0.75,
                    help="keep only this central part of each 224x224 crop (mostly face, little room)")
    ap.add_argument("--out", default="models/tiny")
    ap.add_argument("--balanced", action="store_true",
                    help="weight classes so a big class (e.g. Unknown) does not drown out small ones")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    tf.keras.utils.set_random_seed(SEED)

    common = dict(validation_split=args.val_split, seed=SEED, color_mode="grayscale",
                  image_size=(224, 224), batch_size=args.batch, label_mode="int")
    train_ds = tf.keras.utils.image_dataset_from_directory(args.data, subset="training", **common)
    val_ds = tf.keras.utils.image_dataset_from_directory(args.data, subset="validation", **common)
    class_names = train_ds.class_names
    print("Classes:", class_names)

    # Tight face crop -> SIZE x SIZE. The board must feed the model the same framing:
    # the face filling (almost) the whole square image.
    def tighten(x, y):
        x = tf.image.central_crop(x, args.face_fraction)
        return tf.image.resize(x, (args.size, args.size), method="area"), y
    train_ds = train_ds.map(tighten)
    val_ds = val_ds.map(tighten)

    # Strong augmentation: the board camera gives different framing/lighting than the laptop webcam
    augment = tf.keras.Sequential([
        L.RandomFlip("horizontal"),
        L.RandomRotation(0.1),
        L.RandomZoom((-0.15, 0.1)),
        L.RandomTranslation(0.1, 0.1),
        L.RandomBrightness(0.3, value_range=(0, 255)),
        L.RandomContrast(0.3),
    ])
    # Background suppression: each person was photographed in a different room, so the
    # model could learn the room instead of the face. For each training image we keep
    # the face-shaped ellipse and, at random, either leave the background, fill it with
    # a random grey level, or swap in the background of another image in the batch.
    s = args.size
    yy, xx = np.mgrid[:s, :s]
    ellipse = ((((xx - s / 2) / (0.40 * s)) ** 2 + ((yy - 0.52 * s) / (0.48 * s)) ** 2) <= 1)
    ellipse = tf.constant(ellipse[None, :, :, None], dtype=tf.float32)

    def mask_background(x):
        b = tf.shape(x)[0]
        r = tf.random.uniform([b, 1, 1, 1])
        grey = tf.ones_like(x) * tf.random.uniform([b, 1, 1, 1], 0, 255)
        other = tf.roll(x, shift=1, axis=0)
        bg = tf.where(r < 0.33, x, tf.where(r < 0.66, grey, other))
        return x * ellipse + bg * (1 - ellipse)

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = (train_ds.cache().shuffle(500, seed=SEED)
                .map(lambda x, y: (tf.clip_by_value(augment(mask_background(x), training=True), 0, 255), y),
                     num_parallel_calls=AUTOTUNE)
                .prefetch(AUTOTUNE))
    val_ds = val_ds.cache().prefetch(AUTOTUNE)

    model = build_tiny_model(len(class_names), args.size)
    model.summary()
    model.compile(optimizer=tf.keras.optimizers.Adam(3e-3),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    model_path = os.path.join(args.out, "tiny_face_model.keras")
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(model_path, monitor="val_accuracy", save_best_only=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=6, min_lr=1e-5),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=25, restore_best_weights=True),
    ]
    class_weight = None
    if args.balanced:
        labels = np.concatenate([y.numpy() for _, y in train_ds])
        counts = np.bincount(labels, minlength=len(class_names))
        class_weight = {i: len(labels) / (len(class_names) * c) for i, c in enumerate(counts) if c}
        print("Training images per class:", dict(zip(class_names, counts.tolist())))
        print("Class weights:", {class_names[i]: round(float(w), 2) for i, w in class_weight.items()})
    hist = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks,
                     class_weight=class_weight)

    model.save(model_path)
    model.export(os.path.join(args.out, "saved_model"))
    with open(os.path.join(args.out, "class_names.json"), "w") as f:
        json.dump(class_names, f, indent=2)

    # Curves
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for k, a in (("accuracy", ax[0]), ("loss", ax[1])):
        a.plot(hist.history[k], label="train"); a.plot(hist.history["val_" + k], label="validation")
        a.set_title(k.capitalize()); a.set_xlabel("epoch"); a.legend()
    fig.tight_layout(); fig.savefig(os.path.join(args.out, "tiny_training_curves.png"), dpi=120)
    plt.close(fig)

    # Evaluate
    y_true, y_pred = [], []
    for x, y in val_ds:
        y_true.extend(y.numpy())
        y_pred.extend(np.argmax(model.predict(x, verbose=0), axis=1))
    labels = list(range(len(class_names)))
    print("\nValidation report (tiny float model):")
    print(classification_report(y_true, y_pred, labels=labels, target_names=class_names, zero_division=0))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    ConfusionMatrixDisplay(cm, display_labels=class_names).plot(cmap="Blues")
    plt.title("Tiny model - confusion matrix (validation)")
    plt.savefig(os.path.join(args.out, "tiny_confusion_matrix.png"), dpi=120, bbox_inches="tight")

    acc = float(np.mean(np.array(y_true) == np.array(y_pred)))
    info = {
        "target_board": "Arduino Nano 33 BLE Sense Rev2 (nRF52840, 1 MB flash, 256 KB RAM)",
        "input": {"shape": [1, args.size, args.size, 1], "color": "grayscale",
                  "dtype": "float32", "range": "0-255 (model rescales internally)"},
        "output": {"shape": [1, len(class_names)], "type": "softmax probabilities",
                   "classes": class_names},
        "preprocessing": "square crop where the face fills almost the whole image "
                         "(central %d%% of preprocess.py's 224x224 crop), grayscale, resize to %dx%d"
                         % (round(args.face_fraction * 100), args.size, args.size),
        "params": int(model.count_params()),
        "float32_weights_kb": round(model.count_params() * 4 / 1024, 1),
        "validation_accuracy_float": round(acc, 4),
        "validation_images": len(y_true),
    }
    with open(os.path.join(args.out, "model_info.json"), "w") as f:
        json.dump(info, f, indent=2)
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
