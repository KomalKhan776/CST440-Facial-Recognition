"""
Quantization-aware training (QAT) for the board model.

Why: deep fine-tuning (train_mnet.py --unfreeze 100) gives a float model that scores 89-95 % on the
held-out test set, but post-training int8 quantization drops it to 68-79 % (rounding errors add up
over ~65 layers; Noah collapses to Luke). QAT continues training WITH simulated int8 rounding, so the
weights adapt to it and the int8 model keeps the float accuracy.

Steps:
    1. Load a float model trained by train_mnet.py (Keras 3).
    2. Rebuild the same network in tf_keras as a FLAT graph (tfmot cannot quantize nested models)
       and copy the weights over by layer name; check the two give the same predictions.
    3. Wrap it with tfmot quantize_model (fake-quant on weights + activations, conv+BN folding).
    4. Fine-tune a few epochs at a small learning rate, same data / augmentation as train_mnet.py.
    5. Convert to full-int8 TFLite, evaluate on the held-out test folder, write the C header.

Usage (iter5, the deployed model):
    python train_qat.py --float_dir models/iter5_float --data dataset/v6/train_np --out models/iter5 --balanced
        --test_dir dataset/v4/test --header arduino/face_model_data.h
"""
import argparse
import json
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np
import tensorflow as tf
import keras as keras3                 # Keras 3: reads the .keras files written by train_mnet.py
import tf_keras as tfk                 # Keras 2: what tfmot's QAT works with
import tensorflow_model_optimization as tfmot

from eval_test import evaluate
from quantize import write_c_header, largest_activation_bytes

L = tfk.layers


def build_flat(num_classes, size, alpha):
    """Same network as train_mnet.build(), but with MobileNetV2's layers inlined (no nested model)."""
    inp = tfk.Input(shape=(size, size, 1), name="image")
    x = L.Rescaling(1.0 / 127.5, offset=-1.0, name="rescale")(inp)
    x = L.Concatenate(name="gray3")([x, x, x])
    base = tfk.applications.MobileNetV2(input_tensor=x, alpha=alpha, include_top=False, weights=None)
    x = L.GlobalAveragePooling2D(name="gap")(base.output)
    x = L.Dropout(0.3, name="drop")(x)
    out = L.Dense(num_classes, activation="softmax", name="probs")(x)
    return tfk.Model(inp, out, name="mnet_face_flat")


def keras3_weights_by_name(model):
    """{layer name: weights} for every layer, descending into the nested MobileNetV2."""
    out = {}
    for layer in model.layers:
        if hasattr(layer, "layers"):
            out.update(keras3_weights_by_name(layer))
        elif layer.get_weights():
            out[layer.name] = layer.get_weights()
    return out


def calibrate_ranges(flat, qat, images, batch=32):
    """Set every QAT activation range (output_min/max, pre_activation_min/max) to the real min/max the
    float model produces on the training images. tfmot starts them at -6..6 and moves them with a
    0.999 moving average, far too slowly for MobileNetV2's linear 'project' layers (range ~ +-50),
    so without this the QAT model starts out clipped and training collapses."""
    wrappers = [l for l in qat.layers if any(v.name.split("/")[-1].startswith(("output_m", "pre_activation_m"))
                                            for v in l.non_trainable_variables)]
    inner = [w.layer.name for w in wrappers]
    probe = tfk.Model(flat.input, [flat.get_layer(n).output for n in inner] + [flat.get_layer("gap").output])
    W, b = flat.get_layer("probs").get_weights()
    lo = [np.inf] * (len(inner) + 1); hi = [-np.inf] * (len(inner) + 1)
    for i in range(0, len(images), batch):
        outs = probe.predict(images[i:i + batch], verbose=0)
        outs[-1] = outs[-1] @ W + b                       # logits = Dense input before softmax
        for j, o in enumerate(outs):
            lo[j] = min(lo[j], float(o.min())); hi[j] = max(hi[j], float(o.max()))
    for j, w in enumerate(wrappers):
        name = inner[j]
        mn, mx = (lo[-1], hi[-1]) if name == "probs" else (lo[j], hi[j])
        mn, mx = min(mn, 0.0), max(mx, 0.0)              # range must contain 0
        for v in w.non_trainable_variables:
            short = v.name.split("/")[-1]
            if short.startswith(("output_min", "pre_activation_min")):
                v.assign(mn)
            elif short.startswith(("output_max", "pre_activation_max")):
                v.assign(mx)
    print(f"calibrated activation ranges of {len(wrappers)} QAT layers from {len(images)} images")


def load_ds(data, subset, args):
    ds = keras3.utils.image_dataset_from_directory(
        data, subset=subset, validation_split=args.val_split, seed=args.split_seed,
        color_mode="grayscale", image_size=(224, 224), batch_size=args.batch, label_mode="categorical")
    names = ds.class_names
    ds = ds.map(lambda x, y: (tf.image.resize(tf.image.central_crop(x, args.face_fraction),
                                              (args.size, args.size), method="area"), y))
    return ds, names


def augment_fn(args):
    """Same augmentation as train_mnet.py, written with tf.image ops (no Keras-version mixing)."""
    def aug(x, y):
        b = tf.shape(x)[0]
        x = tf.image.random_flip_left_right(x)
        # small random zoom + shift: crop a random box and resize back
        scale = tf.random.uniform([b], 0.85, 1.0)
        oy = tf.random.uniform([b], 0, 1) * (1 - scale)
        ox = tf.random.uniform([b], 0, 1) * (1 - scale)
        boxes = tf.stack([oy, ox, oy + scale, ox + scale], axis=1)
        x = tf.image.crop_and_resize(x, boxes, tf.range(b), (args.size, args.size))
        x = x + tf.random.uniform([b, 1, 1, 1], -0.35, 0.35) * 255            # brightness
        mean = tf.reduce_mean(x, axis=[1, 2, 3], keepdims=True)
        x = (x - mean) * tf.random.uniform([b, 1, 1, 1], 0.6, 1.4) + mean     # contrast
        small = tf.image.resize(x, (48, 48), method="area")                    # blur like a far face
        blurred = tf.image.resize(small, (args.size, args.size), method="bilinear")
        use = tf.cast(tf.random.uniform([b, 1, 1, 1]) < 0.4, x.dtype)
        x = use * blurred + (1 - use) * x
        x = x + tf.random.normal(tf.shape(x), stddev=tf.random.uniform([], 0, 8))
        return tf.clip_by_value(x, 0, 255), y
    return aug


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--float_dir", required=True, help="folder with tiny_face_model.keras from train_mnet.py")
    ap.add_argument("--data", default="dataset/v3/iter2_processed")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--size", type=int, default=96)
    ap.add_argument("--alpha", type=float, default=0.35)
    ap.add_argument("--face_fraction", type=float, default=0.75)
    ap.add_argument("--val_split", type=float, default=0.2)
    ap.add_argument("--split_seed", type=int, default=42)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ls", type=float, default=0.1)
    ap.add_argument("--balanced", action="store_true")
    ap.add_argument("--test_dir", default=None, help="held-out test folder to report int8 accuracy on")
    ap.add_argument("--thresholds", type=float, nargs="+", default=[0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7])
    ap.add_argument("--header", default=None, help="also write the Arduino C header here")
    ap.add_argument("--board_threshold", type=float, default=0.55,
                    help="Unknown threshold written into the header (0.55 was best over 3 seeds for iter5)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    keras3.utils.set_random_seed(args.seed)
    tfk.utils.set_random_seed(args.seed)

    # ---- 1-2. float model -> flat tf_keras copy ------------------------------------------
    src = keras3.models.load_model(os.path.join(args.float_dir, "tiny_face_model.keras"), compile=False)
    names = json.load(open(os.path.join(args.float_dir, "class_names.json")))
    flat = build_flat(len(names), args.size, args.alpha)
    w = keras3_weights_by_name(src)
    copied = 0
    for layer in flat.layers:
        if layer.get_weights():
            if layer.name not in w:
                raise SystemExit(f"no weights for layer {layer.name} in the float model")
            layer.set_weights(w[layer.name]); copied += 1
    print(f"copied weights into {copied} layers")
    # Keep BatchNorm frozen (inference mode), exactly like train_mnet.py (base called with training=False).
    # Training BN statistics on batches of 16 under fake-quant wrecks the model.
    for layer in flat.layers:
        if isinstance(layer, L.BatchNormalization):
            layer.trainable = False

    train_ds, dnames = load_ds(args.data, "training", args)
    train_ds = train_ds.cache()
    val_ds, _ = load_ds(args.data, "validation", args)
    val_ds = val_ds.cache()
    if dnames != names:
        raise SystemExit(f"dataset classes {dnames} != model classes {names}")
    xb = next(iter(val_ds))[0]
    diff = np.abs(src.predict(xb, verbose=0) - flat.predict(xb, verbose=0)).max()
    print(f"flat copy vs original: max prob difference {diff:.2e}")
    if diff > 1e-3:
        raise SystemExit("weight copy failed - the rebuilt model does not match")

    # ---- 3. quantization-aware wrapper ---------------------------------------------------
    def annotate(layer):
        if isinstance(layer, (L.Rescaling, L.Concatenate, L.InputLayer)):
            return layer                    # no weights; the converter quantizes these anyway
        return tfmot.quantization.keras.quantize_annotate_layer(layer)
    annotated = tfk.models.clone_model(flat, clone_function=annotate)
    annotated.set_weights(flat.get_weights())
    with tfmot.quantization.keras.quantize_scope():
        qat = tfmot.quantization.keras.quantize_apply(annotated)

    # ---- 4. fine-tune with fake-quant ----------------------------------------------------
    batches = list(train_ds)                     # one pass: calibration images + labels
    rep = np.concatenate([x.numpy() for x, _ in batches])
    class_weight = None
    if args.balanced:
        labels = np.concatenate([y.numpy().argmax(1) for _, y in batches])
        counts = np.bincount(labels, minlength=len(names))
        class_weight = {i: len(labels) / (len(names) * c) for i, c in enumerate(counts) if c}
    AUTOTUNE = tf.data.AUTOTUNE
    train_aug = (train_ds.shuffle(1000, seed=args.seed)
                 .map(augment_fn(args), num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE))
    val_c = val_ds.prefetch(AUTOTUNE)

    qat.compile(optimizer=tfk.optimizers.Adam(args.lr),
                loss=tfk.losses.CategoricalCrossentropy(label_smoothing=args.ls), metrics=["accuracy"])
    calibrate_ranges(flat, qat, rep)
    print("QAT model before fine-tuning:", dict(zip(["loss", "acc"], qat.evaluate(val_c, verbose=0))))
    ckpt = os.path.join(args.out, "qat_best.weights.h5")
    h = qat.fit(train_aug, validation_data=val_c, epochs=args.epochs, class_weight=class_weight,
                callbacks=[tfk.callbacks.ModelCheckpoint(ckpt, monitor="val_accuracy", save_best_only=True,
                                                         save_weights_only=True)])
    qat.load_weights(ckpt)

    # ---- 5. int8 conversion --------------------------------------------------------------
    conv = tf.lite.TFLiteConverter.from_keras_model(qat)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = lambda: ([img[None]] for img in rep)
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.int8
    conv.inference_output_type = tf.int8
    tfl = conv.convert()
    tfl_path = os.path.join(args.out, "tiny_face_model_int8.tflite")
    with open(tfl_path, "wb") as f:
        f.write(tfl)
    with open(os.path.join(args.out, "class_names.json"), "w") as f:
        json.dump(names, f, indent=2)

    report = {"float_dir": args.float_dir, "args": vars(args), "int8_tflite_kb": round(len(tfl) / 1024, 1),
              "largest_activation_kb": round(largest_activation_bytes(tfl) / 1024, 1),
              "qat_val_accuracy_history": [round(v, 4) for v in h.history["val_accuracy"]]}
    if args.test_dir:
        report["test_dir"] = args.test_dir
        report["test_float"] = evaluate(os.path.join(args.float_dir, "tiny_face_model.keras"),
                                        args.test_dir, args.thresholds)
        report["test_int8"] = evaluate(tfl_path, args.test_dir, args.thresholds)
        print(f"\nHeld-out test set {args.test_dir} (max prob < threshold -> Unknown):")
        print("threshold   float model   int8 (QAT)")
        for th in report["test_int8"]:
            print(f"  {th:<8} {report['test_float'][th]['overall']:8.1%}     {report['test_int8'][th]['overall']:6.1%}"
                  f"   per class {report['test_int8'][th]['per_class']}")
    if args.header:
        it = tf.lite.Interpreter(model_content=tfl); it.allocate_tensors()
        write_c_header(args.header, tfl, it.get_input_details()[0], it.get_output_details()[0],
                       names, args.board_threshold)
        print("wrote", args.header)
    with open(os.path.join(args.out, "quantization_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"int8 model {report['int8_tflite_kb']} KB -> {tfl_path}")


if __name__ == "__main__":
    main()
