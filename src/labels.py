"""SEED-VII label tables.

来自 Design.md 中的 session_sequences：每个 session 4 folds，每 fold 5 clips。
情绪缩写映射到整数类别索引。
"""
from __future__ import annotations
from typing import Dict, List, Tuple

# 7-class label mapping
EMOTION_LABELS: Dict[str, str] = {
    "H": "Happy", "U": "Surprise", "N": "Neutral",
    "D": "Disgust", "F": "Fear", "S": "Sad", "A": "Anger",
}
EMOTION_TO_IDX: Dict[str, int] = {
    "H": 0, "U": 1, "N": 2, "D": 3, "F": 4, "S": 5, "A": 6,
}
IDX_TO_EMOTION: Dict[int, str] = {v: k for k, v in EMOTION_TO_IDX.items()}
N_CLASSES = 7

# session -> fold -> clip emotion code
SESSION_SEQUENCES: Dict[int, Dict[int, List[str]]] = {
    1: {1: ["H", "N", "D", "S", "A"], 2: ["A", "S", "D", "N", "H"],
        3: ["H", "N", "D", "S", "A"], 4: ["A", "S", "D", "N", "H"]},
    2: {1: ["A", "S", "F", "N", "U"], 2: ["U", "N", "F", "S", "A"],
        3: ["A", "S", "F", "N", "U"], 4: ["U", "N", "F", "S", "A"]},
    3: {1: ["H", "U", "D", "F", "A"], 2: ["A", "F", "D", "U", "H"],
        3: ["H", "U", "D", "F", "A"], 4: ["A", "F", "D", "U", "H"]},
    4: {1: ["D", "S", "F", "U", "H"], 2: ["H", "U", "F", "S", "D"],
        3: ["D", "S", "F", "U", "H"], 4: ["H", "U", "F", "S", "D"]},
}

FOLDS_PER_SESSION = 4
TRIALS_PER_FOLD = 5
TRIALS_PER_SESSION = FOLDS_PER_SESSION * TRIALS_PER_FOLD  # 20


def trial_id_to_emotion(session_id: int, trial_id_1based: int) -> str:
    """trial_id ∈ [1, 20] for a single session, in playback order."""
    if not (1 <= trial_id_1based <= TRIALS_PER_SESSION):
        raise ValueError(f"trial_id must be in [1,{TRIALS_PER_SESSION}], got {trial_id_1based}")
    fold_idx = (trial_id_1based - 1) // TRIALS_PER_FOLD + 1
    in_fold_idx = (trial_id_1based - 1) % TRIALS_PER_FOLD
    return SESSION_SEQUENCES[session_id][fold_idx][in_fold_idx]


def trial_field_to_session_trial(field_id_1based: int) -> Tuple[int, int]:
    """Map .mat field name 1..80 -> (session 1-4, trial 1-20)."""
    if not (1 <= field_id_1based <= 80):
        raise ValueError(f"field_id must be in [1,80], got {field_id_1based}")
    session_id = (field_id_1based - 1) // TRIALS_PER_SESSION + 1
    trial_in_session = (field_id_1based - 1) % TRIALS_PER_SESSION + 1
    return session_id, trial_in_session


def field_id_to_label(field_id_1based: int) -> Tuple[int, str, int]:
    """Convenience: 1..80 -> (session_id, emotion_code, class_idx)."""
    sid, tid = trial_field_to_session_trial(field_id_1based)
    code = trial_id_to_emotion(sid, tid)
    return sid, code, EMOTION_TO_IDX[code]