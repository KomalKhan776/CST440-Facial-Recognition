"""
Evaluate a face model (.keras float or .tflite int8/float) on a held-out test folder.

Preprocessing matches training and the board: grayscale -> 224x224 -> central 75 % -> 96x96 (area).
Unknown rule (same as the board): if max prob < threshold -> "Unknown".
A predicted "NoPerson" stays "NoPerson" (it is not turned into Unknown).

Usage:
    python eval_test.py models/iter5/tiny_face_model_int8.tflite dataset/v4/test --errors
    python eval_test.py models/mnet_iter3/tiny_face_model.keras --thresholds 0.5 0.6 0.7
class_names.json must be next to the model.
"""
import argparse
import json
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import numpy as np
import tensorflow as tf

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def load_test(test_dir, size=96, face_fraction=0.75):
    X, Y, F = [], [], []
    for c in sorted(os.listdir(test_dir)):
        d = os.path.join(test_dir, c)
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if not f.lower().endswith(IMG_EXTS):
                continue
            im = tf.io.decode_image(tf.io.read_file(os.path.join(d, f)), channels=1, expand_animations=False)
            im = tf.image.resize(im, (224, 224))
            im = tf.image.resize(tf.image.central_crop(im, face_fraction), (size, size), method="area")
            X.append(im.numpy()); Y.append(c); F.append(os.path.join(c, f))
    return np.array(X, dtype=np.float32), Y, F


def predict_probs(model_path, X):
    if model_path.endswith(".tflite"):
        it = tf.lite.Interpreter(model_path=model_path)
        it.allocate_tensors()
        inp, out = it.get_input_details()[0], it.get_output_details()[0]
        in_s, in_zp = inp["quantization"]; out_s, out_zp = out["quantization"]
        P = []
        for img in X:
            x = img[None]
            if inp["dtype"] == np.int8:
                x = np.clip(np.round(x / in_s + in_zp), -128, 127).astype(np.int8)
            it.set_tensor(inp["index"], x); it.invoke()
            o = it.get_tensor(out["index"])[0]
            if out["dtype"] == np.int8:
                o = (o.astype(np.float32) - out_zp) * out_s
            P.append(o)
        return np.array(P)
    import keras  # Keras 3 explicitly (tf.keras may point at tf_keras once tfmot is imported)
    return keras.models.load_model(model_path, compile=False).predict(X, verbose=0)


def score(P, Y, class_names, th):
    pred = []
    for p in P:
        i = int(p.argmax())
        pred.append(class_names[i] if (p[i] >= th or class_names[i] == "NoPerson") else "Unknown")
    acc = float(np.mean([a == b for a, b in zip(pred, Y)]))
    per = {c: round(float(np.mean([a == c for a, y in zip(pred, Y) if y == c])), 3) for c in sorted(set(Y))}
    return acc, per, pred


def evaluate(model_path, test_dir, thresholds):
    """{threshold: {overall, per_class, errors}} for one model on one test folder."""
    cn = json.load(open(os.path.join(os.path.dirname(model_path), "class_names.json")))
    X, Y, F = load_test(test_dir)
    P = predict_probs(model_path, X)
    res = {}
    for th in thresholds:
        acc, per, pred = score(P, Y, cn, th)
        res[str(th)] = {"overall": round(acc, 4), "per_class": per,
                        "errors": [f"{f} -> {p}" for f, p, y in zip(F, pred, Y) if p != y]}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("test_dir", nargs="?", default="dataset/v3/test_processed")
    ap.add_argument("--thresholds", type=float, nargs="+", default=[0.5, 0.55, 0.6, 0.7])
    ap.add_argument("--json", default=None, help="write results here")
    ap.add_argument("--errors", action="store_true", help="list misclassified files")
    a = ap.parse_args()
    res = evaluate(a.model, a.test_dir, a.thresholds)
    for th, r in res.items():
        print(th, r["overall"], r["per_class"])
        if a.errors:
            for e in r["errors"]:
                print("    ", e)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
