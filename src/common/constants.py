"""Single source of truth for labels, emotion order, and hard-merge rules.

All label IDs, names, original dataset label mappings, and B2 hard-merge rules
are declared here per `docs/contract.md` Section 1.
NOTE: Do NOT declare `WTRANS_INIT_RULE` here (deferred until after G1 / kappa).
"""

from typing import Dict, Tuple

# ---------------------------------------------------------------------------
# 1. Target 3-class sentiment labels (NEU-ESC & UIT-VSFC)
#    0 = negative, 1 = neutral, 2 = positive
# ---------------------------------------------------------------------------
NEG: int = 0
NEU: int = 1
POS: int = 2

LABELS3: Tuple[str, ...] = ("negative", "neutral", "positive")
LABEL3_TO_ID: Dict[str, int] = {name: idx for idx, name in enumerate(LABELS3)}
ID_TO_LABEL3: Dict[int, str] = {idx: name for idx, name in enumerate(LABELS3)}

# ---------------------------------------------------------------------------
# 2. Original NEU-ESC 4-class labels and mapping to 3-class sentiment
#    Original NEU-ESC: 0 = neutral, 1 = positive, 2 = negative, 3 = toxic
#    Mapping: toxic (3) merges into negative (0)
# ---------------------------------------------------------------------------
NEUESC_ORIG_NEUTRAL: int = 0
NEUESC_ORIG_POSITIVE: int = 1
NEUESC_ORIG_NEGATIVE: int = 2
NEUESC_ORIG_TOXIC: int = 3

NEUESC_ORIG_LABELS: Dict[int, str] = {
    NEUESC_ORIG_NEUTRAL: "neutral",
    NEUESC_ORIG_POSITIVE: "positive",
    NEUESC_ORIG_NEGATIVE: "negative",
    NEUESC_ORIG_TOXIC: "toxic",
}

NEUESC_ORIG_TO_LABEL3: Dict[int, int] = {
    NEUESC_ORIG_NEUTRAL: NEU,   # 0 -> 1 (neutral)
    NEUESC_ORIG_POSITIVE: POS,  # 1 -> 2 (positive)
    NEUESC_ORIG_NEGATIVE: NEG,  # 2 -> 0 (negative)
    NEUESC_ORIG_TOXIC: NEG,     # 3 -> 0 (toxic -> negative)
}

NEUESC_ORIG_NAME_TO_LABEL3: Dict[str, int] = {
    "neutral": NEU,
    "positive": POS,
    "negative": NEG,
    "toxic": NEG,
}

# ---------------------------------------------------------------------------
# 3. Source 7-class emotion labels (UIT-VSMEC)
#    Enjoyment 0, Sadness 1, Fear 2, Anger 3, Disgust 4, Surprise 5, Other 6
# ---------------------------------------------------------------------------
EMO_ENJOYMENT: int = 0
EMO_SADNESS: int = 1
EMO_FEAR: int = 2
EMO_ANGER: int = 3
EMO_DISGUST: int = 4
EMO_SURPRISE: int = 5
EMO_OTHER: int = 6

EMOTIONS7: Tuple[str, ...] = (
    "Enjoyment",
    "Sadness",
    "Fear",
    "Anger",
    "Disgust",
    "Surprise",
    "Other",
)

EMOTION7_TO_ID: Dict[str, int] = {name: idx for idx, name in enumerate(EMOTIONS7)}
ID_TO_EMOTION7: Dict[int, str] = {idx: name for idx, name in enumerate(EMOTIONS7)}

# ---------------------------------------------------------------------------
# 4. Frozen B2 Hard-Merge Rule (7 emotions -> 3 sentiments)
#    Enjoyment -> positive (2)
#    Sadness, Fear, Anger, Disgust -> negative (0)
#    Surprise, Other -> neutral (1)
# ---------------------------------------------------------------------------
HARD_MERGE_RULE: Dict[int, int] = {
    EMO_ENJOYMENT: POS,
    EMO_SADNESS: NEG,
    EMO_FEAR: NEG,
    EMO_ANGER: NEG,
    EMO_DISGUST: NEG,
    EMO_SURPRISE: NEU,
    EMO_OTHER: NEU,
}

HARD_MERGE_RULE_BY_NAME: Dict[str, str] = {
    ID_TO_EMOTION7[emo_id]: ID_TO_LABEL3[sent_id]
    for emo_id, sent_id in HARD_MERGE_RULE.items()
}

# Groupings of emotion indices per target sentiment class (ordered by NEG, NEU, POS)
HARD_MERGE_GROUPS: Dict[int, Tuple[int, ...]] = {
    NEG: (EMO_SADNESS, EMO_FEAR, EMO_ANGER, EMO_DISGUST),
    NEU: (EMO_SURPRISE, EMO_OTHER),
    POS: (EMO_ENJOYMENT,),
}
