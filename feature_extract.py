"""
================================================================================
NeuralStride AI - Leg Raise Classifier: Feature Extraction (self-contained)
================================================================================

Adapted from the main project's feature_utils.py + angle_utils.calculate_angle
so this package has zero dependency on the rest of the NeuralStride repo (no
mediapipe import, no project-relative imports) — Furqan can drop this whole
folder into a different codebase and it will still work.

Two adaptations from the original:
  1. extract_features() no longer takes an `mp_pose` module argument — the
     MediaPipe Pose landmark indices (0-32) are hardcoded constants below
     instead of being read off mp_pose.PoseLandmark.*.value. They are a fixed
     part of the MediaPipe Pose spec, not something that varies by install.
  2. Each landmark can be EITHER a MediaPipe NormalizedLandmark object
     (attribute access: lm.x, lm.y, lm.z, lm.visibility) OR a plain dict
     (lm["x"], lm["y"], lm["z"], lm["visibility"]) — whichever a caller's
     upstream pose source hands over. See _xyzv() below.

Must stay numerically IDENTICAL to feature_utils.py's extract_features() —
this is what makes wrapper predictions match the training-time evaluation.
Do not hand-simplify the math here without checking against feature_utils.py.
================================================================================
"""

import math

# ---------------------------------------------------------------------------
# MediaPipe Pose landmark indices (fixed spec, 0-32) — see landmark_exporter.py
# LANDMARK_NAMES in the main project for the full list; only the ones the
# feature vector depends on are named here.
# ---------------------------------------------------------------------------
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_ELBOW, RIGHT_ELBOW = 13, 14
LEFT_WRIST, RIGHT_WRIST = 15, 16
LEFT_HIP, RIGHT_HIP = 23, 24
LEFT_KNEE, RIGHT_KNEE = 25, 26
LEFT_ANKLE, RIGHT_ANKLE = 27, 28
LEFT_FOOT_INDEX, RIGHT_FOOT_INDEX = 31, 32

FEATURE_NAMES = [
    "left_knee_angle",
    "right_knee_angle",
    "left_hip_angle",
    "right_hip_angle",
    "left_elbow_angle",
    "right_elbow_angle",
    "left_shoulder_angle",
    "right_shoulder_angle",
    "left_ankle_angle",
    "right_ankle_angle",
    "back_lean_angle",
    "knee_angle_diff",
    "hip_angle_diff",
    "elbow_angle_diff",
    "hip_drop",
    "shoulder_tilt",
    "left_knee_toe_offset_x",
    "right_knee_toe_offset_x",
    "hip_height_over_ankle",
    "torso_length",
    "ankle_separation",
    "wrist_separation",
]
FEATURE_COUNT = len(FEATURE_NAMES)  # 22

# Feature-vector index pairs to swap when mirroring a sample left <-> right.
# Not used by predict_rep() (mirroring is a train-time augmentation only) —
# kept here for parity with feature_utils.py in case a future caller wants
# mirrored-input test-time augmentation.
MIRROR_PAIRS = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (16, 17)]
_MIRROR_NEGATE_INDICES = (14, 15)

CORE_LANDMARKS = [11, 12, 23, 24, 25, 26, 27, 28]
CORE_LANDMARKS_LEFT = [11, 23, 25, 27]
CORE_LANDMARKS_RIGHT = [12, 24, 26, 28]
SIDE_VIEWS = ("side", "diagonal")

# For each of the 22 features, the landmark indices it depends on.
# Per-feature confidence = min(visibility) of these landmarks.
FEATURE_LANDMARK_DEPS = [
    [23, 25, 27],
    [24, 26, 28],
    [11, 23, 25],
    [12, 24, 26],
    [11, 13, 15],
    [12, 14, 16],
    [13, 11, 23],
    [14, 12, 24],
    [25, 27, 31],
    [26, 28, 32],
    [11, 12, 23, 24],
    [23, 25, 27, 24, 26, 28],
    [11, 23, 25, 12, 24, 26],
    [11, 13, 15, 12, 14, 16],
    [23, 24],
    [11, 12],
    [25, 31],
    [26, 32],
    [23, 24, 27, 28],
    [11, 12, 23, 24],
    [27, 28],
    [15, 16],
]
assert len(FEATURE_LANDMARK_DEPS) == FEATURE_COUNT

_GEOMETRY_CLAMP_MIN = -10.0
_GEOMETRY_CLAMP_MAX = 10.0
_DEGENERATE_EPS = 1e-6


def _xyzv(lm):
    """(x, y, z, visibility) from either a MediaPipe landmark object
    (attribute access) or a plain dict (key access)."""
    if isinstance(lm, dict):
        return lm["x"], lm["y"], lm["z"], lm["visibility"]
    return lm.x, lm.y, lm.z, lm.visibility


def _visibility(lm):
    return lm["visibility"] if isinstance(lm, dict) else lm.visibility


def _pt(lm):
    """Normalized (x, y) tuple for a landmark — resolution independent."""
    x, y, _, _ = _xyzv(lm)
    return (x, y)


def _midpoint(a, b):
    return ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)


def _euclidean(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _clamp(value):
    if value < _GEOMETRY_CLAMP_MIN:
        return _GEOMETRY_CLAMP_MIN
    if value > _GEOMETRY_CLAMP_MAX:
        return _GEOMETRY_CLAMP_MAX
    return value


def calculate_angle(a, b, c):
    """Angle in degrees at vertex b, formed by points a-b-c (Law of Cosines).
    Identical to angle_utils.calculate_angle in the main project."""
    bax, bay = a[0] - b[0], a[1] - b[1]
    bcx, bcy = c[0] - b[0], c[1] - b[1]
    denom = math.hypot(bax, bay) * math.hypot(bcx, bcy)
    if denom == 0:
        return 0.0
    cosang = (bax * bcx + bay * bcy) / denom
    cosang = max(-1.0, min(1.0, cosang))
    return math.degrees(math.acos(cosang))


def _compute_feature_confidence(landmarks):
    confidence = []
    for deps in FEATURE_LANDMARK_DEPS:
        confidence.append(min(_visibility(landmarks[i]) for i in deps))
    return confidence


def _best_side_landmarks(landmarks):
    left_avg = sum(_visibility(landmarks[i]) for i in CORE_LANDMARKS_LEFT) / len(CORE_LANDMARKS_LEFT)
    right_avg = sum(_visibility(landmarks[i]) for i in CORE_LANDMARKS_RIGHT) / len(CORE_LANDMARKS_RIGHT)
    return CORE_LANDMARKS_LEFT if left_avg >= right_avg else CORE_LANDMARKS_RIGHT


def extract_features(landmarks, camera_view="front"):
    """
    Compute the 22-value biomechanical feature vector for one frame.

    Args:
        landmarks: sequence of 33 landmarks (index 0-32), each either a
                   MediaPipe NormalizedLandmark object or a dict with
                   x/y/z/visibility keys.
        camera_view: "front", "side", or "diagonal" — affects only the
                     quality-score landmark selection, not the features.

    Returns:
        (feature_vector: list[float] len 22, quality_score: float 0-1,
         feature_confidence: list[float] len 22) — same contract as
        feature_utils.extract_features().
    """
    left_shoulder = _pt(landmarks[LEFT_SHOULDER])
    right_shoulder = _pt(landmarks[RIGHT_SHOULDER])
    left_elbow = _pt(landmarks[LEFT_ELBOW])
    right_elbow = _pt(landmarks[RIGHT_ELBOW])
    left_wrist = _pt(landmarks[LEFT_WRIST])
    right_wrist = _pt(landmarks[RIGHT_WRIST])
    left_hip = _pt(landmarks[LEFT_HIP])
    right_hip = _pt(landmarks[RIGHT_HIP])
    left_knee = _pt(landmarks[LEFT_KNEE])
    right_knee = _pt(landmarks[RIGHT_KNEE])
    left_ankle = _pt(landmarks[LEFT_ANKLE])
    right_ankle = _pt(landmarks[RIGHT_ANKLE])
    left_foot_index = _pt(landmarks[LEFT_FOOT_INDEX])
    right_foot_index = _pt(landmarks[RIGHT_FOOT_INDEX])

    hip_width = _euclidean(left_hip, right_hip)
    if hip_width < _DEGENERATE_EPS:
        hip_width = _euclidean(left_shoulder, right_shoulder)
        if hip_width < _DEGENERATE_EPS:
            zero_conf = [0.0] * FEATURE_COUNT
            return [0.0] * FEATURE_COUNT, 0.0, zero_conf

    left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
    right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)
    left_hip_angle = calculate_angle(left_shoulder, left_hip, left_knee)
    right_hip_angle = calculate_angle(right_shoulder, right_hip, right_knee)
    left_elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
    right_elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
    left_shoulder_angle = calculate_angle(left_elbow, left_shoulder, left_hip)
    right_shoulder_angle = calculate_angle(right_elbow, right_shoulder, right_hip)
    left_ankle_angle = calculate_angle(left_knee, left_ankle, left_foot_index)
    right_ankle_angle = calculate_angle(right_knee, right_ankle, right_foot_index)

    mid_hip = _midpoint(left_hip, right_hip)
    mid_shoulder = _midpoint(left_shoulder, right_shoulder)
    mid_ankle = _midpoint(left_ankle, right_ankle)

    vertical_ref = (mid_hip[0], mid_hip[1] - 1.0)
    back_lean_angle = calculate_angle(mid_shoulder, mid_hip, vertical_ref)

    knee_angle_diff = abs(left_knee_angle - right_knee_angle)
    hip_angle_diff = abs(left_hip_angle - right_hip_angle)
    elbow_angle_diff = abs(left_elbow_angle - right_elbow_angle)

    hip_drop = _clamp((left_hip[1] - right_hip[1]) / hip_width)
    shoulder_tilt = _clamp((left_shoulder[1] - right_shoulder[1]) / hip_width)
    left_knee_toe_offset_x = _clamp((left_knee[0] - left_foot_index[0]) / hip_width)
    right_knee_toe_offset_x = _clamp((right_knee[0] - right_foot_index[0]) / hip_width)
    hip_height_over_ankle = _clamp((mid_ankle[1] - mid_hip[1]) / hip_width)
    torso_length = _clamp(_euclidean(mid_shoulder, mid_hip) / hip_width)
    ankle_separation = _clamp(abs(left_ankle[0] - right_ankle[0]) / hip_width)
    wrist_separation = _clamp(abs(left_wrist[0] - right_wrist[0]) / hip_width)

    features = [
        left_knee_angle, right_knee_angle, left_hip_angle, right_hip_angle,
        left_elbow_angle, right_elbow_angle, left_shoulder_angle, right_shoulder_angle,
        left_ankle_angle, right_ankle_angle, back_lean_angle,
        knee_angle_diff, hip_angle_diff, elbow_angle_diff,
        hip_drop, shoulder_tilt, left_knee_toe_offset_x, right_knee_toe_offset_x,
        hip_height_over_ankle, torso_length, ankle_separation, wrist_separation,
    ]

    features = [float(v) if math.isfinite(v) else 0.0 for v in features]
    assert len(features) == FEATURE_COUNT
    assert all(math.isfinite(v) for v in features)

    if camera_view in SIDE_VIEWS:
        quality_landmarks = _best_side_landmarks(landmarks)
    else:
        quality_landmarks = CORE_LANDMARKS
    quality = min(_visibility(landmarks[i]) for i in quality_landmarks)

    feature_confidence = _compute_feature_confidence(landmarks)

    return features, quality, feature_confidence


def mirror_features(vec):
    """Left/right-mirrored copy of a feature vector. Not used by predict_rep
    — kept for parity with feature_utils.py (see MIRROR_PAIRS note above)."""
    mirrored = list(vec)
    for i, j in MIRROR_PAIRS:
        mirrored[i], mirrored[j] = vec[j], vec[i]
    for i in _MIRROR_NEGATE_INDICES:
        mirrored[i] = -vec[i]
    return mirrored
