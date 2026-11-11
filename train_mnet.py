"""
Board-sized TRANSFER-LEARNING model for the Arduino Nano 33 BLE Sense Rev2.

The tiny CNN in train_tiny.py (~20k parameters, trained from scratch) generalises poorly to new
sessions (new room, lighting, distance, head angle). This script instead uses MobileNetV2 at
width 0.35 with ImageNet weights - the same idea as train.py, but small enough for the board:

    input : 96 x 96 x 1 grayscale, pixel values 0-255 (same as train_tiny.py, so the board code,
            quantize.py and the framing rules do not change)
    output: softmax over the class folders (e.g. Komal, Luke, Noah, Reece, Unknown)

The grayscale image is repeated to 3 channels inside the model so the ImageNet weights can be used.
Size: ~400k parameters -> roughly 0.5 MB as int8 (fits the 1 MB flash).

Usage:
    python train_mnet.py --data dataset/v3/iter2_processed --out models/mnet_iter3 --balanced
    python quantize.py --model_dir models/mnet_iter3 --data dataset/v3/iter2_processed
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

SEED = 42
L = tf.keras.layers


def build(num_classes, size, alpha):
    inputs = tf.keras.Input(shape=(size, size, 1), name="image")
    x = L.Rescaling(1.0 / 127.5, offset=-1.0)(inputs)        # 0-255 -> -1..1 (MobileNetV2 range)
    x = L.Concatenate()([x, x, x])                            # grayscale -> 3 channels
    base = tf.keras.applications.MobileNetV2(input_shape=(size, size, 3), alpha=alpha,
                                             include_top=False, weights="imagenet")
    base.trainable = False
    x = base(x, training=False)
    x = L.GlobalAveragePooling2D()(x)
    x = L.Dropout(0.3)(x)
    outputs = L.Dense(num_classes, activation="softmax", name="probs")(x)
    return tf.keras.Model(inputs, outputs, name="mnet_face"), base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset/processed")
    ap.add_argument("--size", type=int, default=96)
    ap.add_argument("--alpha", type=float, default=0.35)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--fine_tune_epochs", type=int, default=25)
    ap.add_argument("--val_split", type=float, default=0.2)
    ap.add_argument("--face_fraction", type=float, default=0.75)
    ap.add_argument("--balanced", action="store_true")
    ap.add_argument("--out", default="models/mnet")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    tf.keras.utils.set_random_seed(SEED)
    common = dict(validation_split=args.val_split, seed=SEED, color_mode="grayscale",
                  image_size=(224, 224), batch_size=args.batch, label_mode="int")
    train_ds = tf.keras.utils.image_dataset_from_directory(args.data, subset="training", **common)
    val_ds = tf.keras.utils.image_dataset_from_directory(args.data, subset="validation", **common)
    class_names = train_ds.class_names
    print("Classes:", class_names)

    # Same framing as train_tiny.py / the board: central 75 % of the face crop, 96x96
    tighten = lambda x, y: (tf.image.resize(tf.image.central_crop(x, args.face_fraction),
                                            (args.size, args.size), method="area"), y)
    train_ds, val_ds = train_ds.map(tighten), val_ds.map(tighten)

    # Augmentation for new sessions: head turns, distance, lighting, a cheap camera
    augment = tf.keras.Sequential([
        L.RandomFlip("horizontal"),
        L.RandomRotation(0.08),
        L.RandomZoom((-0.2, 0.15)),
        L.RandomTranslation(0.1, 0.1),
        L.RandomBrightness(0.35, value_range=(0, 255)),
        L.RandomContrast(0.4),
    ])

    def degrade(x):  # random blur + noise, like a small/far face from the board camera
        b = tf.shape(x)[0]
        small = tf.image.resize(x, (48, 48), method="area")
        blurred = tf.image.resize(small, (args.size, args.size), method="bilinear")
        use = tf.cast(tf.random.uniform([b, 1, 1, 1]) < 0.4, x.dtype)
        x = use * blurred + (1 - use) * x
        return x + tf.random.normal(tf.shape(x), stddev=tf.random.uniform([], 0, 8))

    class_weight = None
    if args.balanced:
        labels = np.concatenate([y.numpy() for _, y in train_ds])
        counts = np.bincount(labels, minlength=len(class_names))
        class_weight = {i: len(labels) / (len(class_names) * c) for i, c in enumerate(counts) if c}
        print("Training images per class:", dict(zip(class_names, counts.tolist())))

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = (train_ds.cache().shuffle(1000, seed=SEED)
                .map(lambda x, y: (tf.clip_by_value(degrade(augment(x, training=True)), 0, 255), y),
                     num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE))
    val_ds = val_ds.cache().prefetch(AUTOTUNE)

    model, base = build(len(class_names), args.size, args.alpha)
    path = os.path.join(args.out, "tiny_face_model.keras")   # same file name quantize.py expects
    cbs = [tf.keras.callbacks.ModelCheckpoint(path, monitor="val_accuracy", save_best_only=True),
           tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)]

    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    h1 = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=cbs,
                   class_weight=class_weight)
    # Fine-tune the top of MobileNetV2 with a small learning rate
    base.trainable = True
    for layer in base.layers[:-40]:
        layer.trainable = False
    model.compile(optimizer=tf.keras.optimizers.Adam(5e-5), loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    h2 = model.fit(train_ds, validation_data=val_ds, epochs=args.fine_tune_epochs, callbacks=cbs,
                   class_weight=class_weight)

    model.save(path)
    with open(os.path.join(args.out, "class_names.json"), "w") as f:
        json.dump(class_names, f, indent=2)

    acc = h1.history["accuracy"] + h2.history["accuracy"]
    vacc = h1.history["val_accuracy"] + h2.history["val_accuracy"]
    plt.plot(acc, label="train"); plt.plot(vacc, label="validation")
    plt.axvline(len(h1.history["accuracy"]) - 0.5, color="grey", ls="--", lw=1)
    plt.title("MobileNetV2-0.35 (96x96 gray) accuracy"); plt.xlabel("epoch"); plt.legend()
    plt.savefig(os.path.join(args.out, "training_curves.png"), dpi=120, bbox_inches="tight")

    _, val_acc = model.evaluate(val_ds, verbose=0)
    info = {"architecture": f"MobileNetV2 alpha={args.alpha}, ImageNet weights, 96x96 gray->3ch",
            "input": {"shape": [1, args.size, args.size, 1], "range": "0-255 (model rescales internally)"},
            "classes": class_names, "params": int(model.count_params()),
            "validation_accuracy_float": round(float(val_acc), 4),
            "epochs_run": [len(h1.history["accuracy"]), len(h2.history["accuracy"])]}
    with open(os.path.join(args.out, "model_info.json"), "w") as f:
        json.dump(info, f, indent=2)
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
