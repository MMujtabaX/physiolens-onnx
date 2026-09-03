"""
================================================================================
PhysioLens - Leg Raise Classifier (ONNX, TensorFlow-free inference)
================================================================================

Same public interface and same preprocessing contract as
legraise_classifier/classifier.py's LegRaiseClassifier, but the model forward
pass runs on `onnxruntime` instead of TensorFlow/Keras -- this is the piece
that lets the deployed app drop the TensorFlow dependency entirely (see
convert_to_onnx.py for how legraise_cnn_lstm.onnx was produced from the
original .keras model, and its printed max-difference check for numeric
parity).

PREPROCESSING CONTRACT -- identical to legraise_classifier/classifier.py,
copied here (not imported) so this module has zero dependency outside
onnxruntime + numpy + feature_extract.py (also copied, itself TF-free):
    1. Per-frame 22-feature vector + quality + 22-confidence vector via
       feature_extract.extract_features().
    2. Drop frames with quality < QUALITY_FLOOR (0.5); raise if fewer than
       MIN_VALID_FRAMES (8) remain.
    3. Pad/truncate the remaining (T, 22) sequence to WINDOW_LEN (32) --
       centered truncate if longer, end-pad with zeros if shorter.
    4. Normalize the 22 features with the TRAIN-fit mean/std from
       legraise_norm_stats.npz (never refit here). Confidence is NOT
       normalized, matching training.
    5. Concatenate normalized features + raw confidence -> (32, 44).
    6. Model outputs LOGITS (not probabilities) -- same as the Keras model,
       since the ONNX graph is a direct trace of it. Sigmoid is applied by
       hand here, after the forward pass.
================================================================================
"""

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort

from feature_extract import extract_features

PACKAGE_DIR = Path(__file__).resolve().parent

QUALITY_FLOOR = 0.5
MIN_VALID_FRAMES = 8


def _pad_or_truncate(feat, conf, window_len):
    """Identical to build_dataset.py / classifier.py's pad_or_truncate()."""
    t = feat.shape[0]
    if t == window_len:
        return feat, conf, np.ones(window_len, dtype=np.float32)
    if t > window_len:
        excess = t - window_len
        start = excess // 2
        end = start + window_len
        return feat[start:end], conf[start:end], np.ones(window_len, dtype=np.float32)
    pad_len = window_len - t
    feat_p = np.vstack([feat, np.zeros((pad_len, feat.shape[1]))])
    conf_p = np.vstack([conf, np.zeros((pad_len, conf.shape[1]))])
    mask = np.concatenate([np.ones(t, dtype=np.float32), np.zeros(pad_len, dtype=np.float32)])
    return feat_p, conf_p, mask


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


class LegRaiseClassifier:
    """
    Drop-in replacement for legraise_classifier.LegRaiseClassifier with the
    same predict_rep() contract, backed by onnxruntime instead of Keras.

    Usage:
        clf = LegRaiseClassifier()
        result = clf.predict_rep(rep_frames)
        session_summary = aggregate_session([result, ...])
    """

    def __init__(self, model_dir=None, threshold_knee=0.5, threshold_shallow=0.5):
        model_dir = Path(model_dir) if model_dir else PACKAGE_DIR

        with open(model_dir / "label_names.json") as f:
            cfg = json.load(f)
        self.label_codes = cfg["label_codes"]
        self.window_len = cfg["window_len"]
        if "KNEE_BENT" not in self.label_codes or "SHALLOW" not in self.label_codes:
            raise ValueError(f"Expected KNEE_BENT and SHALLOW in label_codes, got {self.label_codes}")
        self._idx = {code: i for i, code in enumerate(self.label_codes)}

        norm = np.load(model_dir / "legraise_norm_stats.npz")
        self.mean = norm["mean"].astype(np.float64)
        self.std = norm["std"].astype(np.float64)

        self.session = ort.InferenceSession(
            str(model_dir / "legraise_cnn_lstm.onnx"),
            providers=["CPUExecutionProvider"],
        )

        # Identify which ONNX graph input is "features" ((*, 32, 44), rank 3)
        # vs "mask" ((*, 32), rank 2) by shape rank rather than by name --
        # tf2onnx does not always preserve the original Keras input names
        # verbatim, but the two inputs are unambiguous by rank.
        self._features_input_name = None
        self._mask_input_name = None
        for inp in self.session.get_inputs():
            if len(inp.shape) == 3:
                self._features_input_name = inp.name
            elif len(inp.shape) == 2:
                self._mask_input_name = inp.name
        if self._features_input_name is None or self._mask_input_name is None:
            raise RuntimeError(
                f"Could not identify features/mask inputs from ONNX graph inputs: "
                f"{[(i.name, i.shape) for i in self.session.get_inputs()]}"
            )
        self._output_name = self.session.get_outputs()[0].name

        self.threshold_knee = threshold_knee
        self.threshold_shallow = threshold_shallow

    def predict_rep(self, rep_frames, camera_view="front"):
        """Same signature, same return dict shape as
        legraise_classifier.classifier.LegRaiseClassifier.predict_rep()."""
        if not rep_frames:
            raise ValueError("predict_rep() called with an empty rep_frames list")

        feats, confs = [], []
        for landmarks in rep_frames:
            feat, quality, conf = extract_features(landmarks, camera_view=camera_view)
            if quality < QUALITY_FLOOR:
                continue
            feats.append(feat)
            confs.append(conf)

        if len(feats) < MIN_VALID_FRAMES:
            raise ValueError(
                f"Only {len(feats)}/{len(rep_frames)} frames passed the quality floor "
                f"({QUALITY_FLOOR}) -- need >= {MIN_VALID_FRAMES} valid frames to score a rep."
            )

        feat_arr = np.array(feats, dtype=np.float64)
        conf_arr = np.array(confs, dtype=np.float64)

        feat_padded, conf_padded, mask = _pad_or_truncate(feat_arr, conf_arr, self.window_len)

        feat_norm = (feat_padded - self.mean) / self.std
        model_input = np.concatenate([feat_norm, conf_padded], axis=-1).astype(np.float32)

        logits = self.session.run(
            [self._output_name],
            {
                self._features_input_name: model_input[None, ...],
                self._mask_input_name: mask[None, ...].astype(np.float32),
            },
        )[0][0]
        probs = _sigmoid(logits)

        p_knee = float(probs[self._idx["KNEE_BENT"]])
        p_shallow = float(probs[self._idx["SHALLOW"]])

        knee_flag = p_knee >= self.threshold_knee
        shallow_flag = p_shallow >= self.threshold_shallow

        return {
            "faults": {
                "knee_bent": knee_flag,
                "shallow": shallow_flag,
            },
            "fault_probabilities": {
                "knee_bent": round(p_knee, 4),
                "shallow": round(p_shallow, 4),
            },
            "fault_percentages": {
                "knee_bent": round(p_knee * 100, 1),
                "shallow": round(p_shallow * 100, 1),
            },
            "rep_quality": round((1 - p_knee) * (1 - p_shallow), 4),
            "rep_quality_percent": round((1 - p_knee) * (1 - p_shallow) * 100, 1),
            "predicted_clean": (not knee_flag) and (not shallow_flag),
        }


def aggregate_session(rep_results):
    """Identical to legraise_classifier.classifier.aggregate_session() --
    pure aggregation, no I/O, no model calls."""
    n = len(rep_results)
    if n == 0:
        return {
            "total_reps": 0,
            "avg_rep_quality_percent": 0.0,
            "avg_fault_percent": {"knee_bent": 0.0, "shallow": 0.0},
            "flagged_reps": {"knee_bent": 0, "shallow": 0},
            "clean_reps": 0,
        }

    avg_quality = sum(r["rep_quality_percent"] for r in rep_results) / n
    avg_knee = sum(r["fault_percentages"]["knee_bent"] for r in rep_results) / n
    avg_shallow = sum(r["fault_percentages"]["shallow"] for r in rep_results) / n
    flagged_knee = sum(1 for r in rep_results if r["faults"]["knee_bent"])
    flagged_shallow = sum(1 for r in rep_results if r["faults"]["shallow"])
    clean = sum(1 for r in rep_results if r["predicted_clean"])

    return {
        "total_reps": n,
        "avg_rep_quality_percent": round(avg_quality, 1),
        "avg_fault_percent": {
            "knee_bent": round(avg_knee, 1),
            "shallow": round(avg_shallow, 1),
        },
        "flagged_reps": {
            "knee_bent": flagged_knee,
            "shallow": flagged_shallow,
        },
        "clean_reps": clean,
    }
