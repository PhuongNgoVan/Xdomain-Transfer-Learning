"""CSV column schemas and keys defined in `docs/contract.md` Section 2."""

from typing import Tuple

from src.common.constants import EMOTIONS7

# ---------------------------------------------------------------------------
# 2.1. Split file schema (`data/splits/neuesc_*.csv`)
# ---------------------------------------------------------------------------
SPLIT_COLUMNS: Tuple[str, ...] = (
    "id",
    "text",
    "label",
    "subset",
)

VALID_SUBSETS: Tuple[str, ...] = ("train", "dev", "test")

# ---------------------------------------------------------------------------
# 2.2. Per-sample 3-class prediction file schema (`results/predictions/*.csv`)
# ---------------------------------------------------------------------------
PRED_COLUMNS: Tuple[str, ...] = (
    "id",
    "y_true",
    "y_pred",
    "logit_neg",
    "logit_neu",
    "logit_pos",
)

# ---------------------------------------------------------------------------
# 2.3. Unified result file schema (`results/{member}.csv`)
# ---------------------------------------------------------------------------
RESULT_COLUMNS: Tuple[str, ...] = (
    "run_id",
    "model",
    "variant",
    "dataset",
    "split",
    "inference_mode",
    "fraction",
    "seed",
    "lr",
    "tau",
    "loss",
    "precision",
    "max_length",
    "best_step",
    "n_train",
    "prior_neg",
    "prior_neu",
    "prior_pos",
    "macro_f1",
    "accuracy",
    "weighted_f1",
    "balanced_acc",
    "f1_neg",
    "f1_neu",
    "f1_pos",
    "ece",
    "gpu",
    "code_version",
    "data_hash",
    "config_path",
    "official",
)

RESULT_KEY_COLUMNS: Tuple[str, ...] = (
    "run_id",
    "dataset",
    "split",
    "inference_mode",
)

VALID_INFERENCE_MODES: Tuple[str, ...] = ("plain", "prior")
VALID_TARGET_DATASETS: Tuple[str, ...] = ("neuesc_dev", "neuesc_test", "vsfc_test")

# ---------------------------------------------------------------------------
# 2.4. Stage 1 result file schema (`results/stage1.csv`)
#      f1_{7 emotions} derived dynamically from EMOTIONS7 in lowercase
# ---------------------------------------------------------------------------
STAGE1_F1_COLUMNS: Tuple[str, ...] = tuple(
    f"f1_{emo.lower()}" for emo in EMOTIONS7
)

STAGE1_COLUMNS: Tuple[str, ...] = (
    "run_id",
    "backbone",
    "seed",
    "lr",
    "best_step",
    "macro_f1",
    "accuracy",
    "weighted_f1",
    *STAGE1_F1_COLUMNS,
    "precision",
    "gpu",
    "code_version",
    "data_hash",
    "config_path",
)

# ---------------------------------------------------------------------------
# 2.7. 7-class logit prediction file schema (`results/predictions7/*.csv`)
# ---------------------------------------------------------------------------
PRED7_LOGIT_COLUMNS: Tuple[str, ...] = tuple(
    f"logit_{i}" for i in range(len(EMOTIONS7))
)

PRED7_COLUMNS: Tuple[str, ...] = (
    "id",
    "y_true7",
    "y_pred7",
    *PRED7_LOGIT_COLUMNS,
)

# ---------------------------------------------------------------------------
# 2.8. Polarity annotation sample & key schemas (`data/clean/polarity_*.csv`)
# ---------------------------------------------------------------------------
POLARITY_SAMPLE_COLUMNS: Tuple[str, ...] = (
    "id",
    "text",
    "polarity_label",
)

POLARITY_KEY_COLUMNS: Tuple[str, ...] = (
    "id",
    "text",
    "emotion_id",
    "emotion_name",
)

