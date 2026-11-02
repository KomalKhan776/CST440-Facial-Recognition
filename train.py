"""
Step 2 - Train the face recognition model.

Uses transfer learning: a MobileNetV2 network pre-trained on ImageNet is used
as a feature extractor, and a small new classifier head is trained on top to
recognise the people in dataset/processed/<PersonName>/.

Outputs (in models/):
    face_model.keras       - the trained model
    class_names.json       - list of person names (label order)
    training_curves.png    - accuracy / loss plots
    confusion_matrix.png   - results on the validation set

Usage:
    python train.py
    python train.py --epochs 20 --fine_tune_epochs 10
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


def build_model(num_classes, img_size):
    base = tf.keras.applications.MobileNetV2(
        input_shape=(img_size, img_size, 3), include_top=False, weights="imagenet"
    )
    base.trainable = False

    augment = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.08),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomBrightness(0.2),
        tf.keras.layers.RandomContrast(0.2),
    ], name="augmentation")

    inputs = tf.keras.Input(shape=(img_size, img_size, 3))
    x = augment(inputs)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    return tf.keras.Model(inputs, outputs), base


def plot_history(histories, path):
    acc, val_acc, loss, val_loss = [], [], [], []
    for h in histories:
        acc += h.history["accuracy"]; val_acc += h.history["val_accuracy"]
        loss += h.history["loss"]; val_loss += h.history["val_loss"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(acc, label="train"); ax[0].plot(val_acc, label="validation")
    ax[0].set_title("Accuracy"); ax[0].set_xlabel("epoch"); ax[0].legend()
    ax[1].plot(loss, label="train"); ax[1].plot(val_loss, label="validation")
    ax[1].set_title("Loss"); ax[1].set_xlabel("epoch"); ax[1].legend()
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset/processed")
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--fine_tune_epochs", type=int, default=10)
    ap.add_argument("--val_split", type=float, default=0.2)
    ap.add_argument("--out", default="models")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    tf.keras.utils.set_random_seed(SEED)

    common = dict(
        validation_split=args.val_split, seed=SEED,
        image_size=(args.size, args.size), batch_size=args.batch, label_mode="int",
    )
    train_ds = tf.keras.utils.image_dataset_from_directory(args.data, subset="training", **common)
    val_ds = tf.keras.utils.image_dataset_from_directory(args.data, subset="validation", **common)
    class_names = train_ds.class_names
    print("Classes:", class_names)

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(500, seed=SEED).prefetch(AUTOTUNE)
    val_ds = val_ds.cache().prefetch(AUTOTUNE)

    model, base = build_model(len(class_names), args.size)
    model_path = os.path.join(args.out, "face_model.keras")
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(model_path, monitor="val_accuracy", save_best_only=True),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
    ]

    # Phase 1: train only the new classifier head
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.summary()
    h1 = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks)
    histories = [h1]

    # Phase 2: fine-tune the top layers of MobileNetV2 with a small learning rate
    if args.fine_tune_epochs > 0:
        base.trainable = True
        for layer in base.layers[:-30]:
            layer.trainable = False
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
                      loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        h2 = model.fit(train_ds, validation_data=val_ds,
                       epochs=args.fine_tune_epochs, callbacks=callbacks)
        histories.append(h2)

    model.save(model_path)
    with open(os.path.join(args.out, "class_names.json"), "w") as f:
        json.dump(class_names, f, indent=2)
    plot_history(histories, os.path.join(args.out, "training_curves.png"))

    # Evaluate on validation set
    y_true, y_pred = [], []
    for x, y in val_ds:
        y_true.extend(y.numpy())
        y_pred.extend(np.argmax(model.predict(x, verbose=0), axis=1))
    print("\nValidation report:")
    print(classification_report(y_true, y_pred, labels=list(range(len(class_names))),
                                target_names=class_names, zero_division=0))
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    ConfusionMatrixDisplay(cm, display_labels=class_names).plot(cmap="Blues")
    plt.title("Confusion matrix (validation)")
    plt.savefig(os.path.join(args.out, "confusion_matrix.png"), dpi=120, bbox_inches="tight")

    print(f"\nModel saved to {model_path}")


if __name__ == "__main__":
    main()
