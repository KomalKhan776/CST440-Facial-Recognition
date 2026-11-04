"""
Quantize the tiny face model to full int8 TensorFlow Lite for the Arduino Nano 33 BLE Sense Rev2.

Steps:
    1. Load the float model made by train_tiny.py (models/tiny/tiny_face_model.keras).
    2. Load dataset/processed/ with EXACTLY the same split and preprocessing as train_tiny.py
       (grayscale 224x224 -> central 75 % crop -> area-resize to 96x96, values 0-255).
    3. Calibrate int8 ranges with the TRAINING images only (the representative dataset),
       so the validation images stay unseen for an honest accuracy number.
    4. Convert to full-integer int8 (int8 input, int8 output, int8 weights/activations).
    5. Compare float vs int8 accuracy on the validation split and on all images, plus the
       background tests from test_tiny.py, and check against the team targets
       (iteration 1 >= 80 %, iteration 2 >= 95 %).
    6. Export the model as a C array for the Arduino sketch.

Outputs:
    models/tiny/tiny_face_model_int8.tflite     the quantized model
    models/tiny/tiny_face_model_float.tflite    float TFLite, for size/accuracy comparison
    models/tiny/int8_confusion_matrix.png       validation confusion matrix of the int8 model
    models/tiny/quantization_report.json        sizes, accuracies, quantization parameters
    arduino/face_model_data.h                   C array + input/output quantization constants

On the board, feeding a grayscale pixel p (0-255) to the int8 model:
    q = round(p / INPUT_SCALE) + INPUT_ZERO_POINT     (clamped to -128..127)
With the usual calibration (scale ~= 1, zero point = -128) this is simply q = p - 128.
Probabilities out: prob = (q_out - OUTPUT_ZERO_POINT) * OUTPUT_SCALE.

Usage:
    python quantize.py
    python quantize.py --target 0.95
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix

SEED = 42  # must match train_tiny.py so the train/validation split is identical


def load_split(data_dir, subset, size, face_fraction, val_split):
    """Same loading + preprocessing as train_tiny.py (without augmentation)."""
    ds = tf.keras.utils.image_dataset_from_directory(
        data_dir, subset=subset, validation_split=val_split, seed=SEED,
        color_mode="grayscale", image_size=(224, 224), batch_size=32, label_mode="int")
    names = ds.class_names
    ds = ds.map(lambda x, y: (tf.image.resize(tf.image.central_crop(x, face_fraction),
                                              (size, size), method="area"), y))
    X, y = [], []
    for xb, yb in ds:
        X.append(xb.numpy()); y.append(yb.numpy())
    return np.concatenate(X).astype(np.float32), np.concatenate(y), names


def convert(model, rep_images, int8):
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    if int8:
        conv.optimizations = [tf.lite.Optimize.DEFAULT]

        def representative():
            for img in rep_images:
                yield [img[None, ...]]
        conv.representative_dataset = representative
        conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        conv.inference_input_type = tf.int8
        conv.inference_output_type = tf.int8
    return conv.convert()


class TFLiteModel:
    """Runs a .tflite model one image at a time, handling int8 (de)quantization."""

    def __init__(self, tflite_bytes):
        self.it = tf.lite.Interpreter(model_content=tflite_bytes)
        self.it.allocate_tensors()
        self.inp = self.it.get_input_details()[0]
        self.out = self.it.get_output_details()[0]

    def predict(self, X):
        probs = []
        in_scale, in_zp = self.inp["quantization"]
        out_scale, out_zp = self.out["quantization"]
        for img in X:
            x = img[None, ...]
            if self.inp["dtype"] == np.int8:
                x = np.clip(np.round(x / in_scale + in_zp), -128, 127).astype(np.int8)
            self.it.set_tensor(self.inp["index"], x)
            self.it.invoke()
            o = self.it.get_tensor(self.out["index"])[0]
            if self.out["dtype"] == np.int8:
                o = (o.astype(np.float32) - out_zp) * out_scale
            probs.append(o)
        return np.array(probs)


def largest_activation_bytes(tflite_bytes):
    """Rough RAM estimate: biggest single non-constant tensor (the arena needs more than this)."""
    it = tf.lite.Interpreter(model_content=tflite_bytes)
    it.allocate_tensors()
    best = 0
    for t in it.get_tensor_details():
        shape = t["shape"]
        if len(shape) == 4 and shape[0] == 1:  # activations are [1, H, W, C]
            best = max(best, int(np.prod(shape)) * np.dtype(t["dtype"]).itemsize)
    return best


def background_tests(predict, X, y, size):
    """Same face-vs-background checks as test_tiny.py."""
    yy, xx = np.mgrid[:size, :size]
    face = ((xx - size / 2) / (0.40 * size)) ** 2 + ((yy - 0.52 * size) / (0.48 * size)) ** 2 <= 1
    acc = lambda imgs: float(np.mean(predict(imgs).argmax(1) == y))
    res = {"original": acc(X)}
    for v, name in ((0, "black"), (128, "grey"), (255, "white")):
        B = X.copy(); B[:, ~face] = v
        res["background_" + name] = acc(B)
    rng = np.random.default_rng(0)
    S = []
    for i in range(len(X)):
        j = rng.choice(np.where(y != y[i])[0])
        b = X[j].copy(); b[face] = X[i][face]; S.append(b)
    res["face_on_other_room"] = acc(np.stack(S))
    return res


def write_c_header(path, tflite_bytes, inp, out, class_names):
    in_scale, in_zp = inp["quantization"]
    out_scale, out_zp = out["quantization"]
    lines = [
        "// Generated by quantize.py - do not edit by hand.",
        "// Tiny face recognition model, full int8, for TensorFlow Lite Micro.",
        "#pragma once",
        "#include <stdint.h>",
        "",
        f"// Input: [1, {inp['shape'][1]}, {inp['shape'][2]}, {inp['shape'][3]}] int8 grayscale.",
        "// q = round(pixel / FACE_MODEL_INPUT_SCALE) + FACE_MODEL_INPUT_ZERO_POINT, clamp to [-128, 127]",
        f"const float FACE_MODEL_INPUT_SCALE = {in_scale:.9g}f;",
        f"const int FACE_MODEL_INPUT_ZERO_POINT = {in_zp};",
        "// Output: int8 scores; prob = (q - FACE_MODEL_OUTPUT_ZERO_POINT) * FACE_MODEL_OUTPUT_SCALE",
        f"const float FACE_MODEL_OUTPUT_SCALE = {out_scale:.9g}f;",
        f"const int FACE_MODEL_OUTPUT_ZERO_POINT = {out_zp};",
        "",
        f"const int FACE_MODEL_NUM_CLASSES = {len(class_names)};",
        "const char* const FACE_MODEL_CLASSES[] = {" + ", ".join(f'"{n}"' for n in class_names) + "};",
        "",
        f"const unsigned int face_model_len = {len(tflite_bytes)};",
        "alignas(16) const unsigned char face_model[] = {",
    ]
    for i in range(0, len(tflite_bytes), 12):
        lines.append("  " + ", ".join(f"0x{b:02x}" for b in tflite_bytes[i:i + 12]) + ",")
    lines.append("};")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default="models/tiny")
    ap.add_argument("--data", default="dataset/processed")
    ap.add_argument("--size", type=int, default=96)
    ap.add_argument("--face_fraction", type=float, default=0.75)
    ap.add_argument("--val_split", type=float, default=0.2)
    ap.add_argument("--header", default="arduino/face_model_data.h")
    ap.add_argument("--target", type=float, default=0.80,
                    help="accuracy the int8 model must reach (iteration 1: 0.80, iteration 2: 0.95)")
    ap.add_argument("--calibration", choices=["faces", "naive"], default="faces",
                    help="faces = training images (proper); naive = random noise (baseline experiment)")
    ap.add_argument("--naive_images", type=int, default=5)
    args = ap.parse_args()

    # Keep the baseline's files apart so it never overwrites the deployable model.
    tag = "" if args.calibration == "faces" else "_naive"
    if tag:
        root, ext = os.path.splitext(args.header)
        args.header = root + tag + ext

    if not os.path.isdir(args.data):
        raise SystemExit(f"{args.data} not found - download the photos from the team Google Drive "
                         "into dataset/raw/<Name>/ and run: python preprocess.py --per_person 40")

    model = tf.keras.models.load_model(os.path.join(args.model_dir, "tiny_face_model.keras"))
    names = json.load(open(os.path.join(args.model_dir, "class_names.json")))

    Xtr, ytr, dnames = load_split(args.data, "training", args.size, args.face_fraction, args.val_split)
    Xva, yva, _ = load_split(args.data, "validation", args.size, args.face_fraction, args.val_split)
    if dnames != names:
        raise SystemExit(f"dataset folders {dnames} do not match model classes {names} - retrain first")
    print(f"Calibration (training) images: {len(Xtr)}   validation images: {len(Xva)}")

    # ---- Convert -------------------------------------------------------------
    float_tfl = convert(model, None, int8=False)
    if args.calibration == "naive":
        # Baseline for the iteration log: calibrate on a few random-noise images instead of
        # real faces, so int8 ranges do not match real activations. Report it as the naive baseline.
        rng = np.random.default_rng(SEED)
        rep = rng.uniform(0, 255, size=(args.naive_images, args.size, args.size, 1)).astype(np.float32)
        print(f"NAIVE calibration: {args.naive_images} random-noise images (baseline, not for deployment)")
    else:
        rep = Xtr
    int8_tfl = convert(model, rep, int8=True)
    with open(os.path.join(args.model_dir, "tiny_face_model_float.tflite"), "wb") as f:
        f.write(float_tfl)
    with open(os.path.join(args.model_dir, "tiny_face_model_int8" + tag + ".tflite"), "wb") as f:
        f.write(int8_tfl)
    q = TFLiteModel(int8_tfl)

    # ---- Evaluate ------------------------------------------------------------
    Xall, yall = np.concatenate([Xtr, Xva]), np.concatenate([ytr, yva])
    keras_pred = lambda X: model.predict(X, verbose=0)
    p_float_va, p_int8_va = keras_pred(Xva).argmax(1), q.predict(Xva).argmax(1)
    p_float_all, p_int8_all = keras_pred(Xall).argmax(1), q.predict(Xall).argmax(1)

    res = {
        "val_accuracy_float": float(np.mean(p_float_va == yva)),
        "val_accuracy_int8": float(np.mean(p_int8_va == yva)),
        "all_accuracy_float": float(np.mean(p_float_all == yall)),
        "all_accuracy_int8": float(np.mean(p_int8_all == yall)),
        "float_int8_agreement_all": float(np.mean(p_float_all == p_int8_all)),
    }
    bg_float = background_tests(keras_pred, Xall, yall, args.size)
    bg_int8 = background_tests(q.predict, Xall, yall, args.size)

    print("\n                         float     int8")
    print(f"validation accuracy     {res['val_accuracy_float']:6.1%}   {res['val_accuracy_int8']:6.1%}"
          f"   ({len(Xva)} unseen images)")
    print(f"all images accuracy     {res['all_accuracy_float']:6.1%}   {res['all_accuracy_int8']:6.1%}")
    for k in bg_float:
        print(f"{k:<24}{bg_float[k]:6.1%}   {bg_int8[k]:6.1%}")
    print(f"float/int8 agree on {res['float_int8_agreement_all']:.1%} of images")

    print("\nValidation report (int8 model):")
    labels = list(range(len(names)))
    print(classification_report(yva, p_int8_va, labels=labels, target_names=names, zero_division=0))
    cm = confusion_matrix(yva, p_int8_va, labels=labels)
    ConfusionMatrixDisplay(cm, display_labels=names).plot(cmap="Blues")
    plt.title("Tiny model int8 - confusion matrix (validation)")
    plt.savefig(os.path.join(args.model_dir, "int8_confusion_matrix" + tag + ".png"), dpi=120, bbox_inches="tight")
    plt.close()

    # ---- Export for the Arduino ---------------------------------------------
    os.makedirs(os.path.dirname(args.header) or ".", exist_ok=True)
    write_c_header(args.header, int8_tfl, q.inp, q.out, names)

    act = largest_activation_bytes(int8_tfl)
    report = {
        "float_tflite_kb": round(len(float_tfl) / 1024, 1),
        "int8_tflite_kb": round(len(int8_tfl) / 1024, 1),
        "size_reduction": round(1 - len(int8_tfl) / len(float_tfl), 3),
        "largest_activation_kb": round(act / 1024, 1),
        "suggested_tensor_arena_kb": int(np.ceil(act * 3 / 1024 / 8) * 8),
        "input_quantization": {"scale": float(q.inp["quantization"][0]),
                               "zero_point": int(q.inp["quantization"][1])},
        "output_quantization": {"scale": float(q.out["quantization"][0]),
                                "zero_point": int(q.out["quantization"][1])},
        "calibration_images": int(len(Xtr)),
        "validation_images": int(len(Xva)),
        **{k: round(v, 4) for k, v in res.items()},
        "background_tests_float": {k: round(v, 4) for k, v in bg_float.items()},
        "background_tests_int8": {k: round(v, 4) for k, v in bg_int8.items()},
        "target": args.target,
        "meets_target": res["val_accuracy_int8"] >= args.target,
    }
    with open(os.path.join(args.model_dir, "quantization_report" + tag + ".json"), "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nfloat TFLite {report['float_tflite_kb']} KB -> int8 TFLite {report['int8_tflite_kb']} KB "
          f"({report['size_reduction']:.0%} smaller)")
    print(f"largest activation {report['largest_activation_kb']} KB -> "
          f"start with a tensor arena of ~{report['suggested_tensor_arena_kb']} KB")
    print(f"input  scale {report['input_quantization']['scale']:.6g}  zero point "
          f"{report['input_quantization']['zero_point']}")
    print(f"wrote {args.header}")
    verdict = "PASS" if report["meets_target"] else "FAIL"
    print(f"\nint8 validation accuracy {res['val_accuracy_int8']:.1%} vs target {args.target:.0%}: {verdict}")


if __name__ == "__main__":
    main()
