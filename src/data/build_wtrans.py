"""Stage B of M7 (CV-P3): Verify adjudicated disagreements, apply decision rules, and build W_trans.

Per `tasks/M7_polarity_kappa_wtrans (1).md`:
- Verifies SHA-256 of `polarity_nhat.csv`, `polarity_long.csv`, and `polarity_key.csv`
  against `results/polarity/polarity_files_hash.json` from Stage A.
- Verifies `results/polarity/polarity_disagreements.csv` has not been altered in
  `id, text, emotion_name, label_nhat, label_long`, matches the exact set of disagreed IDs,
  and has valid `final_label` in `{0, 1, 2}` for every row.
- Computes overall Cohen's kappa from the independent annotations (never from `final_label`).
- Applies the pre-registered/approved/default decision rules to `Surprise` and `Other`
  while keeping the remaining 5 emotions fixed per `HARD_MERGE_RULE`.
- Exports `data/wtrans/wtrans.csv` (3x7) and `data/wtrans/wtrans_hardmerge.csv` (3x7),
  plus `docs/wtrans_rule.md` using a mandatory `--date YYYY-MM-DD` CLI argument.
- Declares `WTRANS_INIT_RULE` in `src/common/constants.py` ONLY when `docs/wtrans_approval.md`
  is present and valid (`approved: true`).
"""

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from src.common.constants import (
    EMO_ANGER,
    EMO_DISGUST,
    EMO_ENJOYMENT,
    EMO_FEAR,
    EMO_OTHER,
    EMO_SADNESS,
    EMO_SURPRISE,
    EMOTIONS7,
    HARD_MERGE_RULE,
    ID_TO_EMOTION7,
    ID_TO_LABEL3,
    LABELS3,
    NEG,
    NEU,
    POS,
)
from src.common.io import write_csv_schema
from src.common.schema import POLARITY_DISAGREEMENT_COLUMNS
from src.data.polarity_kappa import (
    VALID_POLARITIES,
    VALID_POLARITY_STRS,
    compute_subset_metrics,
    extract_disagreements_df,
    fmt6,
    load_and_validate_polarity_inputs,
    round6,
    wilson_ci,
)

DEFAULT_KAPPA_THRESHOLD: float = 0.40
DEFAULT_POLE_THRESHOLD: float = 0.60
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

EMO_ID_TO_CONST_NAME: Dict[int, str] = {
    EMO_ENJOYMENT: "EMO_ENJOYMENT",
    EMO_SADNESS: "EMO_SADNESS",
    EMO_FEAR: "EMO_FEAR",
    EMO_ANGER: "EMO_ANGER",
    EMO_DISGUST: "EMO_DISGUST",
    EMO_SURPRISE: "EMO_SURPRISE",
    EMO_OTHER: "EMO_OTHER",
}


def validate_date_str(date_str: str) -> str:
    """Validate that `date_str` matches `YYYY-MM-DD`."""
    if not isinstance(date_str, str) or not DATE_RE.match(date_str.strip()):
        raise ValueError(
            f"Invalid --date {date_str!r}; expected format YYYY-MM-DD (e.g. 2026-11-07)."
        )
    return date_str.strip()


def _parse_key_value_md(path: Path) -> Dict[str, str]:
    """Parse `key: value` lines from a markdown/text file."""
    kv: Dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or line.startswith("```"):
            continue
        if ":" in line:
            k, v = line.split(":", 1)
            key = k.strip().strip("-*` ")
            val = v.strip().strip("` ")
            if key:
                kv[key] = val
    return kv


def parse_wtrans_approval(
    approval_path: Optional[Union[str, Path]] = "docs/wtrans_approval.md",
    prereg_path: Optional[Union[str, Path]] = "docs/prereg.md",
) -> Dict[str, Any]:
    """Check `docs/wtrans_approval.md` (and fallback `docs/prereg.md` or defaults).

    Required keys in `docs/wtrans_approval.md`:
        approved: true
        kappa_threshold: 0.40
        pole_threshold: 0.60
        approver: <name>
        date: <YYYY-MM-DD>
    """
    required_keys = {
        "approved",
        "kappa_threshold",
        "pole_threshold",
        "approver",
        "date",
    }

    if approval_path is not None and Path(approval_path).is_file():
        kv = _parse_key_value_md(Path(approval_path))
        if required_keys.issubset(kv.keys()):
            approved_str = kv["approved"].strip().lower()
            approver = kv["approver"].strip()
            date_val = kv["date"].strip()
            try:
                k_thresh = float(kv["kappa_threshold"])
                p_thresh = float(kv["pole_threshold"])
                valid_nums = (0.0 < k_thresh <= 1.0) and (0.0 < p_thresh <= 1.0)
            except ValueError:
                k_thresh = DEFAULT_KAPPA_THRESHOLD
                p_thresh = DEFAULT_POLE_THRESHOLD
                valid_nums = False

            valid_approver = bool(approver) and not (
                approver.startswith("<") and approver.endswith(">")
            )
            valid_date = bool(DATE_RE.match(date_val))
            is_approved = (
                approved_str == "true"
                and valid_nums
                and valid_approver
                and valid_date
            )
            if is_approved:
                return {
                    "is_valid": True,
                    "approved": True,
                    "kappa_threshold": k_thresh,
                    "pole_threshold": p_thresh,
                    "approver": approver,
                    "date": date_val,
                    "source": "wtrans_approval.md",
                }

    # Check prereg.md if present for pre-registered thresholds (without approval)
    if prereg_path is not None and Path(prereg_path).is_file():
        kv_pre = _parse_key_value_md(Path(prereg_path))
        if "kappa_threshold" in kv_pre and "pole_threshold" in kv_pre:
            try:
                k_thresh = float(kv_pre["kappa_threshold"])
                p_thresh = float(kv_pre["pole_threshold"])
                if (0.0 < k_thresh <= 1.0) and (0.0 < p_thresh <= 1.0):
                    return {
                        "is_valid": False,
                        "approved": False,
                        "kappa_threshold": k_thresh,
                        "pole_threshold": p_thresh,
                        "approver": "chưa xác nhận (prereg)",
                        "date": "",
                        "source": "prereg.md",
                    }
            except ValueError:
                pass

    return {
        "is_valid": False,
        "approved": False,
        "kappa_threshold": DEFAULT_KAPPA_THRESHOLD,
        "pole_threshold": DEFAULT_POLE_THRESHOLD,
        "approver": "mặc định (chưa có wtrans_approval.md hợp lệ)",
        "date": "",
        "source": "mặc định",
    }


def decide_emotion_column_rule(
    kappa_all: float,
    counts: Dict[int, int],
    kappa_threshold: float = DEFAULT_KAPPA_THRESHOLD,
    pole_threshold: float = DEFAULT_POLE_THRESHOLD,
    expected_n: int = 50,
    labels: Sequence[int] = VALID_POLARITIES,
) -> Dict[str, Any]:
    """Apply Section 6 decision table to a single emotion column (Surprise or Other).

    Rules:
    - If `total != expected_n` (50): raise ValueError.
    - If `kappa_all` is NaN: raise ValueError (stop and report).
    - Compare `round(kappa_all, 9) >= round(kappa_threshold, 9)`.
    - Compare largest pole integer count `>= ceil(round(pole_threshold * expected_n, 9))`
      (which is `>= 30` for `pole_threshold=0.60, expected_n=50`).
    """
    if kappa_all is None or math.isnan(float(kappa_all)):
        raise ValueError(
            "Overall Cohen's kappa is NaN; stopping per decision table (human decision required)."
        )

    total = sum(int(counts.get(k, 0)) for k in labels)
    if total != expected_n:
        raise ValueError(
            f"Expected exactly {expected_n} sentences for emotion column, got {total}."
        )

    clean_counts = {int(k): int(counts.get(k, 0)) for k in labels}
    proportions = {k: float(clean_counts[k]) / float(total) for k in labels}

    max_count = max(clean_counts.values())
    top_poles = [k for k in labels if clean_counts[k] == max_count]
    max_pole = top_poles[0]

    min_pole_count = int(math.ceil(round(float(pole_threshold) * float(total), 9)))
    is_kappa_high = round(float(kappa_all), 9) >= round(float(kappa_threshold), 9)

    if not is_kappa_high:
        rule = "HARD_MERGE"
        vector = tuple(1.0 if k == NEU else 0.0 for k in labels)
    elif max_count >= min_pole_count:
        if len(top_poles) > 1:
            raise ValueError(
                f"Tie for largest pole at count={max_count} across poles={top_poles}."
            )
        rule = "HARD"
        vector = tuple(1.0 if k == max_pole else 0.0 for k in labels)
    else:
        rule = "SOFT"
        vector = tuple(proportions[k] for k in labels)

    ci_low, ci_high = wilson_ci(max_count, total)
    warn_wilson_pole = bool(
        round(ci_low, 9) <= round(float(pole_threshold), 9) <= round(ci_high, 9)
    )
    warn_wilson_050 = bool(round(ci_low, 9) <= 0.50 <= round(ci_high, 9))

    return {
        "rule": rule,
        "vector": vector,
        "counts": clean_counts,
        "proportions": proportions,
        "total": total,
        "max_pole": int(max_pole),
        "max_pole_name": ID_TO_LABEL3[int(max_pole)],
        "max_count": int(max_count),
        "min_pole_count": int(min_pole_count),
        "wilson_ci_low": ci_low,
        "wilson_ci_high": ci_high,
        "warn_wilson_pole_threshold": warn_wilson_pole,
        "warn_wilson_050": warn_wilson_050,
    }


def verify_stage_b_integrity_and_build_final_labels(
    nhat_path: Union[str, Path],
    long_path: Union[str, Path],
    key_path: Optional[Union[str, Path]],
    disagreements_path: Union[str, Path],
    hash_path: Union[str, Path],
    expected_n_per_emotion: int = 50,
) -> Tuple[pd.DataFrame, Dict[str, Optional[str]], List[str]]:
    """Verify Stage A hashes and disagreement integrity, returning merged DataFrame with `final_label`."""
    p_hash = Path(hash_path)
    if not p_hash.is_file():
        raise FileNotFoundError(
            f"Missing Stage A hash file: {p_hash}. Run polarity_kappa.py first."
        )
    p_dis = Path(disagreements_path)
    if not p_dis.is_file():
        raise FileNotFoundError(
            f"Missing disagreements file: {p_dis}. Run polarity_kappa.py first."
        )

    stored_hashes: Dict[str, Optional[str]] = json.loads(
        p_hash.read_text(encoding="utf-8")
    )

    merged, current_hashes, input_warnings = load_and_validate_polarity_inputs(
        nhat_path=nhat_path,
        long_path=long_path,
        key_path=key_path,
        expected_n_per_emotion=expected_n_per_emotion,
    )

    # 1. Check SHA-256 integrity against Stage A
    for fname in ("polarity_nhat.csv", "polarity_long.csv", "polarity_key.csv"):
        if stored_hashes.get(fname) != current_hashes.get(fname):
            raise ValueError(
                f"SHA-256 integrity check failed for {fname}: "
                f"expected {stored_hashes.get(fname)}, got {current_hashes.get(fname)}."
            )

    # 2. Check polarity_disagreements.csv against actual disagreements
    df_dis = pd.read_csv(p_dis, dtype=str, keep_default_na=False)
    missing_cols = [
        c for c in POLARITY_DISAGREEMENT_COLUMNS if c not in df_dis.columns
    ]
    if missing_cols:
        raise ValueError(
            f"[polarity_disagreements.csv] Missing required columns: {missing_cols}"
        )

    expected_dis = extract_disagreements_df(merged)
    if len(df_dis) != len(expected_dis):
        raise ValueError(
            f"[polarity_disagreements.csv] Row count mismatch: "
            f"expected {len(expected_dis)} disagreed rows, got {len(df_dis)}."
        )

    if len(expected_dis) > 0:
        try:
            df_dis_ids = [int(str(x).strip()) for x in df_dis["id"]]
        except ValueError as exc:
            raise ValueError(
                "[polarity_disagreements.csv] Non-integer id found."
            ) from exc

        df_dis = df_dis.copy()
        df_dis["id_int"] = df_dis_ids
        if df_dis["id_int"].duplicated().any():
            raise ValueError("[polarity_disagreements.csv] Duplicate id found.")
        df_dis = df_dis.sort_values("id_int", kind="mergesort").reset_index(drop=True)

        exp_ids = expected_dis["id"].astype(int).tolist()
        if df_dis["id_int"].tolist() != exp_ids:
            raise ValueError(
                f"[polarity_disagreements.csv] Disagreed IDs do not match actual "
                f"disagreements between annotators: expected {exp_ids}, got {df_dis['id_int'].tolist()}."
            )

        # Check that text, emotion_name, label_nhat, label_long were not altered
        for idx in range(len(expected_dis)):
            rid = exp_ids[idx]
            if str(df_dis.loc[idx, "text"]) != str(expected_dis.loc[idx, "text"]):
                raise ValueError(
                    f"[polarity_disagreements.csv] Modified 'text' at id={rid}."
                )
            if str(df_dis.loc[idx, "emotion_name"]).strip() != str(
                expected_dis.loc[idx, "emotion_name"]
            ).strip():
                raise ValueError(
                    f"[polarity_disagreements.csv] Modified 'emotion_name' at id={rid}."
                )
            if str(df_dis.loc[idx, "label_nhat"]).strip() != str(
                expected_dis.loc[idx, "label_nhat"]
            ).strip():
                raise ValueError(
                    f"[polarity_disagreements.csv] Modified 'label_nhat' at id={rid}."
                )
            if str(df_dis.loc[idx, "label_long"]).strip() != str(
                expected_dis.loc[idx, "label_long"]
            ).strip():
                raise ValueError(
                    f"[polarity_disagreements.csv] Modified 'label_long' at id={rid}."
                )

        # Validate final_label in {0, 1, 2}, non-empty
        bad_final: List[Tuple[int, str]] = []
        adjudicated_map: Dict[int, int] = {}
        for idx in range(len(df_dis)):
            rid = int(df_dis.loc[idx, "id_int"])
            fval = str(df_dis.loc[idx, "final_label"]).strip()
            if fval not in VALID_POLARITY_STRS:
                bad_final.append((rid, fval))
            else:
                adjudicated_map[rid] = int(fval)

        if bad_final:
            raise ValueError(
                f"[polarity_disagreements.csv] Invalid or empty final_label (must be in {{0, 1, 2}}): "
                f"{bad_final[:10]}"
            )
    else:
        adjudicated_map = {}

    final_labels: List[int] = []
    for _, row in merged.iterrows():
        rid = int(row["id"])
        ln = int(row["label_nhat"])
        ll = int(row["label_long"])
        if ln == ll:
            final_labels.append(ln)
        else:
            final_labels.append(adjudicated_map[rid])

    merged_out = merged.copy()
    merged_out["final_label"] = pd.Series(final_labels, dtype=int)
    return merged_out, current_hashes, input_warnings


def build_wtrans_matrix_dict(
    surprise_vector: Tuple[float, float, float],
    other_vector: Tuple[float, float, float],
) -> Dict[int, Tuple[float, float, float]]:
    """Construct mapping `emotion_id -> (p_neg, p_neu, p_pos)` for all 7 emotions."""
    rule_map: Dict[int, Tuple[float, float, float]] = {}
    for emo_id in range(len(EMOTIONS7)):
        if emo_id == EMO_SURPRISE:
            vec = tuple(float(x) for x in surprise_vector)
        elif emo_id == EMO_OTHER:
            vec = tuple(float(x) for x in other_vector)
        else:
            target_pol = HARD_MERGE_RULE[emo_id]
            vec = tuple(1.0 if p == target_pol else 0.0 for p in VALID_POLARITIES)

        col_sum = sum(vec)
        if abs(col_sum - 1.0) > 1e-9:
            raise ValueError(
                f"Column {ID_TO_EMOTION7[emo_id]} does not sum to 1.0: sum={col_sum}"
            )
        rule_map[emo_id] = (vec[0], vec[1], vec[2])
    return rule_map


def wtrans_dict_to_dataframe(
    rule_map: Dict[int, Tuple[float, float, float]],
) -> pd.DataFrame:
    """Convert `emotion_id -> (p_neg, p_neu, p_pos)` to a 3x7 DataFrame for CSV export."""
    rows: List[Dict[str, str]] = []
    for pol_idx, pol_name in enumerate(LABELS3):
        row_dict: Dict[str, str] = {"label": pol_name}
        for emo_id, emo_name in enumerate(EMOTIONS7):
            row_dict[emo_name] = fmt6(rule_map[emo_id][pol_idx])
        rows.append(row_dict)
    return pd.DataFrame(rows, columns=["label", *EMOTIONS7])


def format_wtrans_init_rule_code(
    rule_map: Dict[int, Tuple[float, float, float]],
) -> str:
    """Format the Python declaration of `WTRANS_INIT_RULE` for `src/common/constants.py`."""
    lines = [
        "# ---------------------------------------------------------------------------",
        "# 5. W_trans Initialization Rule (7 emotions -> 3 sentiment probabilities)",
        "#    Declared after G1 / kappa approval per M7 (CV-P3)",
        "# ---------------------------------------------------------------------------",
        "WTRANS_INIT_RULE: Dict[int, Tuple[float, float, float]] = {",
    ]
    for emo_id in range(len(EMOTIONS7)):
        const_name = EMO_ID_TO_CONST_NAME[emo_id]
        v0, v1, v2 = rule_map[emo_id]
        lines.append(f"    {const_name}: ({fmt6(v0)}, {fmt6(v1)}, {fmt6(v2)}),")
    lines.append("}")
    return "\n".join(lines) + "\n"


def update_constants_with_wtrans_rule(
    constants_path: Union[str, Path],
    rule_map: Dict[int, Tuple[float, float, float]],
) -> None:
    """Idempotently declare or update `WTRANS_INIT_RULE` in `constants.py`."""
    p_const = Path(constants_path)
    content = p_const.read_text(encoding="utf-8").replace("\r\n", "\n")
    block = format_wtrans_init_rule_code(rule_map)

    marker = "# ---------------------------------------------------------------------------\n# 5. W_trans Initialization Rule"
    if marker in content:
        prefix = content.split(marker, 1)[0].rstrip() + "\n\n"
        new_content = prefix + block
    else:
        m = re.search(r"^WTRANS_INIT_RULE\s*[:=]", content, flags=re.MULTILINE)
        if m:
            prefix = content[: m.start()].rstrip() + "\n\n"
            new_content = prefix + block
        else:
            new_content = content.rstrip() + "\n\n" + block

    p_const.write_text(new_content, encoding="utf-8", newline="\n")


def render_wtrans_rule_md(
    date_str: str,
    approval_info: Dict[str, Any],
    metrics_by_subset: Dict[str, Dict[str, Any]],
    decisions: Dict[str, Dict[str, Any]],
    wtrans_df: pd.DataFrame,
    file_hashes: Dict[str, Optional[str]],
    warnings_list: List[str],
    proposed_code: str,
    adjudicator: str = "Nhật & Long (hoặc người thứ ba)",
) -> str:
    """Render deterministic `docs/wtrans_rule.md` report."""
    m_all = metrics_by_subset["all"]
    m_other = metrics_by_subset["other"]
    m_surp = metrics_by_subset["surprise"]

    lines: List[str] = [
        "# Quy tắc chuyển 7 cảm xúc sang 3 cực tính (`W_trans` – CV-P3)",
        "",
        f"- **Ngày chốt (`--date`):** `{date_str}`",
        f"- **Nguồn ngưỡng quyết định:** `{approval_info['source']}`",
        f"- **Trạng thái phê duyệt (`wtrans_approval.md`):** `{'Đã phê duyệt (approved: true)' if approval_info['is_valid'] else 'Chưa phê duyệt (dùng ngưỡng ' + approval_info['source'] + ')'}`",
        f"- **Người phê duyệt ngưỡng:** `{approval_info['approver']}`",
        f"- **Người chốt nhãn bất đồng (`final_label`):** `{adjudicator}`",
        f"- **Ngưỡng Cohen's κ toàn bộ (`kappa_threshold`):** `{fmt6(approval_info['kappa_threshold'])}`",
        f"- **Ngưỡng tỉ lệ cực lớn nhất (`pole_threshold`):** `{fmt6(approval_info['pole_threshold'])}` (số đếm nguyên tối thiểu: `{decisions['Surprise']['min_pole_count']}/{decisions['Surprise']['total']}`)",
        "",
        "## 1. Độ đồng thuận trên nhãn độc lập (Giai đoạn A)",
        "",
        "| Tập mẫu | Số câu (n) | po | pe | Cohen's κ | 95% CI (Bootstrap) | Bootstrap bị bỏ | κ trọng số tuyến tính | Gwet AC1 | PABAK |",
        "| --- | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: |",
    ]

    for key, title in (
        ("all", "Toàn bộ (All)"),
        ("other", "Other"),
        ("surprise", "Surprise"),
    ):
        m = metrics_by_subset[key]
        ci_s = f"[{fmt6(m['ci_low'])}, {fmt6(m['ci_high'])}]"
        drop_s = f"{m['bootstrap_dropped_count']}/{m['n_boot']} ({fmt6(m['bootstrap_dropped_rate'])})"
        lines.append(
            f"| {title} | {m['n']} | {fmt6(m['po'])} | {fmt6(m['pe'])} | {fmt6(m['kappa'])} | "
            f"{ci_s} | {drop_s} | {fmt6(m['kappa_linear'])} | {fmt6(m['gwet_ac1'])} | {fmt6(m['pabak'])} |"
        )

    lines.extend(
        [
            "",
            "## 2. Phân bố nhãn cuối (`final_label`) và luật chọn cho từng cột",
            "",
            "| Cảm xúc | Số câu | Đếm (neg, neu, pos) | Tỉ lệ (neg, neu, pos) | Cực lớn nhất | Wilson 95% CI (cực lớn nhất) | Luật được chọn | Vectơ cột (neg, neu, pos) |",
            "| --- | ---: | --- | --- | --- | --- | --- | --- |",
        ]
    )

    for emo_name in ("Surprise", "Other"):
        d = decisions[emo_name]
        c_tup = f"({d['counts'][NEG]}, {d['counts'][NEU]}, {d['counts'][POS]})"
        p_tup = f"({fmt6(d['proportions'][NEG])}, {fmt6(d['proportions'][NEU])}, {fmt6(d['proportions'][POS])})"
        v_tup = f"({fmt6(d['vector'][0])}, {fmt6(d['vector'][1])}, {fmt6(d['vector'][2])})"
        w_ci = f"[{fmt6(d['wilson_ci_low'])}, {fmt6(d['wilson_ci_high'])}]"
        lines.append(
            f"| {emo_name} | {d['total']} | {c_tup} | {p_tup} | "
            f"{d['max_pole_name']} ({d['max_count']}/{d['total']}) | {w_ci} | **{d['rule']}** | `{v_tup}` |"
        )

    lines.extend(
        [
            "",
            "## 3. Ma trận `W_trans` (3 × 7, cột tổng bằng 1)",
            "",
            "| label | " + " | ".join(EMOTIONS7) + " |",
            "| --- | " + " | ".join("---:" for _ in EMOTIONS7) + " |",
        ]
    )

    for _, r in wtrans_df.iterrows():
        vals = " | ".join(str(r[emo]) for emo in EMOTIONS7)
        lines.append(f"| {r['label']} | {vals} |")

    lines.extend(
        [
            "",
            "## 4. SHA-256 các file nhãn đầu vào",
            "",
            f"- `polarity_nhat.csv`: `{file_hashes.get('polarity_nhat.csv')}`",
            f"- `polarity_long.csv`: `{file_hashes.get('polarity_long.csv')}`",
            f"- `polarity_key.csv`: `{file_hashes.get('polarity_key.csv')}`",
            "",
            "## 5. Cảnh báo",
            "",
        ]
    )

    if warnings_list:
        for w in warnings_list:
            lines.append(f"- {w}")
    else:
        lines.append("- Không có cảnh báo.")

    lines.extend(
        [
            "",
            "## 6. Nội dung khai báo `WTRANS_INIT_RULE`",
            "",
            "```python",
            proposed_code.rstrip(),
            "```",
        ]
    )

    return "\n".join(lines) + "\n"


def run_build_wtrans_stage_b(
    date: str,
    nhat_path: Union[str, Path] = "data/clean/polarity_nhat.csv",
    long_path: Union[str, Path] = "data/clean/polarity_long.csv",
    key_path: Optional[Union[str, Path]] = "data/clean/polarity_key.csv",
    disagreements_path: Union[str, Path] = "results/polarity/polarity_disagreements.csv",
    hash_path: Union[str, Path] = "results/polarity/polarity_files_hash.json",
    approval_path: Optional[Union[str, Path]] = "docs/wtrans_approval.md",
    prereg_path: Optional[Union[str, Path]] = "docs/prereg.md",
    out_wtrans_dir: Union[str, Path] = "data/wtrans",
    out_rule_doc: Union[str, Path] = "docs/wtrans_rule.md",
    constants_path: Optional[Union[str, Path]] = "src/common/constants.py",
    expected_n_per_emotion: int = 50,
    n_boot: int = 2000,
    seed: int = 42,
    adjudicator: str = "Nhật & Long (hoặc người thứ ba)",
) -> Dict[str, Any]:
    """Execute Stage B of M7: integrity verification, rule decision, and W_trans export."""
    date_str = validate_date_str(date)

    merged_final, file_hashes, input_warnings = (
        verify_stage_b_integrity_and_build_final_labels(
            nhat_path=nhat_path,
            long_path=long_path,
            key_path=key_path,
            disagreements_path=disagreements_path,
            hash_path=hash_path,
            expected_n_per_emotion=expected_n_per_emotion,
        )
    )

    approval_info = parse_wtrans_approval(
        approval_path=approval_path, prereg_path=prereg_path
    )
    kappa_threshold = float(approval_info["kappa_threshold"])
    pole_threshold = float(approval_info["pole_threshold"])

    # Compute independent kappa metrics (NEVER using final_label)
    sub_other = merged_final[merged_final["emotion_id"] == EMO_OTHER]
    sub_surprise = merged_final[merged_final["emotion_id"] == EMO_SURPRISE]

    m_all = compute_subset_metrics(
        merged_final["label_nhat"].tolist(),
        merged_final["label_long"].tolist(),
        n_boot=n_boot,
        seed=seed,
    )
    m_other = compute_subset_metrics(
        sub_other["label_nhat"].tolist(),
        sub_other["label_long"].tolist(),
        n_boot=n_boot,
        seed=seed,
    )
    m_surprise = compute_subset_metrics(
        sub_surprise["label_nhat"].tolist(),
        sub_surprise["label_long"].tolist(),
        n_boot=n_boot,
        seed=seed,
    )

    # Final polarity counts for Surprise and Other
    surp_counts = {
        k: int((sub_surprise["final_label"] == k).sum()) for k in VALID_POLARITIES
    }
    other_counts = {
        k: int((sub_other["final_label"] == k).sum()) for k in VALID_POLARITIES
    }

    decision_surprise = decide_emotion_column_rule(
        kappa_all=m_all["kappa"],
        counts=surp_counts,
        kappa_threshold=kappa_threshold,
        pole_threshold=pole_threshold,
        expected_n=expected_n_per_emotion,
    )
    decision_other = decide_emotion_column_rule(
        kappa_all=m_all["kappa"],
        counts=other_counts,
        kappa_threshold=kappa_threshold,
        pole_threshold=pole_threshold,
        expected_n=expected_n_per_emotion,
    )

    warnings_list: List[str] = list(input_warnings)
    if approval_info["source"] == "mặc định":
        warnings_list.append(
            "đang dùng ngưỡng mặc định (kappa_threshold=0.40, pole_threshold=0.60) do chưa có wtrans_approval.md hợp lệ"
        )

    for sub_name, m in (("all", m_all), ("other", m_other), ("surprise", m_surprise)):
        if m["warning_ci_biased"]:
            warnings_list.append(
                f"CI có thể lệch trên tập {sub_name}: tỉ lệ bootstrap bị bỏ = {fmt6(m['bootstrap_dropped_rate'])} (> 0.05)"
            )

    if not math.isnan(m_other["kappa"]) and not math.isnan(m_surprise["kappa"]):
        other_high = round(float(m_other["kappa"]), 9) >= round(kappa_threshold, 9)
        surp_high = round(float(m_surprise["kappa"]), 9) >= round(kappa_threshold, 9)
        if other_high != surp_high:
            warnings_list.append(
                f"hai κ riêng nằm hai phía của {fmt6(kappa_threshold)}: "
                f"Other κ={fmt6(m_other['kappa'])}, Surprise κ={fmt6(m_surprise['kappa'])}"
            )

    if decision_surprise["rule"] == "HARD_MERGE" or decision_other["rule"] == "HARD_MERGE":
        warnings_list.append(
            "HARD_MERGE khi κ thấp là lựa chọn bảo thủ (không đủ đồng thuận để tin cực tính); "
            "cực tính của Other và Surprise không ổn định, không có nghĩa hai nhãn này thực sự trung tính"
        )

    for emo_name, dec in (("Surprise", decision_surprise), ("Other", decision_other)):
        if dec["warn_wilson_pole_threshold"]:
            warnings_list.append(
                f"kết luận yếu ({emo_name}): Wilson 95% CI [{fmt6(dec['wilson_ci_low'])}, {fmt6(dec['wilson_ci_high'])}] "
                f"của cực lớn nhất ({dec['max_pole_name']}) chứa ngưỡng {fmt6(pole_threshold)}"
            )
        if dec["warn_wilson_050"]:
            warnings_list.append(
                f"cực lớn nhất có thể không chiếm đa số ({emo_name}): Wilson 95% CI "
                f"[{fmt6(dec['wilson_ci_low'])}, {fmt6(dec['wilson_ci_high'])}] chứa 0.50"
            )

    # Build W_trans and W_trans_hardmerge
    wtrans_map = build_wtrans_matrix_dict(
        surprise_vector=decision_surprise["vector"],
        other_vector=decision_other["vector"],
    )
    hardmerge_map = build_wtrans_matrix_dict(
        surprise_vector=(0.0, 1.0, 0.0),
        other_vector=(0.0, 1.0, 0.0),
    )

    wtrans_df = wtrans_dict_to_dataframe(wtrans_map)
    hardmerge_df = wtrans_dict_to_dataframe(hardmerge_map)

    wtrans_dir_path = Path(out_wtrans_dir)
    wtrans_dir_path.mkdir(parents=True, exist_ok=True)
    wtrans_cols = ("label", *EMOTIONS7)

    wtrans_csv_path = write_csv_schema(
        wtrans_df, wtrans_dir_path / "wtrans.csv", wtrans_cols
    )
    hardmerge_csv_path = write_csv_schema(
        hardmerge_df, wtrans_dir_path / "wtrans_hardmerge.csv", wtrans_cols
    )

    proposed_code = format_wtrans_init_rule_code(wtrans_map)

    rule_doc_path = Path(out_rule_doc)
    rule_doc_path.parent.mkdir(parents=True, exist_ok=True)
    rule_md = render_wtrans_rule_md(
        date_str=date_str,
        approval_info=approval_info,
        metrics_by_subset={"all": m_all, "other": m_other, "surprise": m_surprise},
        decisions={"Surprise": decision_surprise, "Other": decision_other},
        wtrans_df=wtrans_df,
        file_hashes=file_hashes,
        warnings_list=warnings_list,
        proposed_code=proposed_code,
        adjudicator=adjudicator,
    )
    rule_doc_path.write_text(rule_md, encoding="utf-8", newline="\n")

    constants_updated = False
    if approval_info["is_valid"] and constants_path is not None:
        update_constants_with_wtrans_rule(constants_path, wtrans_map)
        constants_updated = True

    return {
        "approval_info": approval_info,
        "decisions": {
            "Surprise": decision_surprise,
            "Other": decision_other,
        },
        "wtrans_map": wtrans_map,
        "hardmerge_map": hardmerge_map,
        "wtrans_df": wtrans_df,
        "hardmerge_df": hardmerge_df,
        "wtrans_csv_path": wtrans_csv_path,
        "hardmerge_csv_path": hardmerge_csv_path,
        "rule_doc_path": rule_doc_path,
        "constants_updated": constants_updated,
        "proposed_code": proposed_code,
        "warnings": warnings_list,
        "metrics": {
            "all": m_all,
            "other": m_other,
            "surprise": m_surprise,
        },
    }


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Stage B (CV-P3): Build W_trans matrix after disagreement adjudication."
    )
    parser.add_argument(
        "--date",
        type=str,
        required=True,
        help="Mandatory date string YYYY-MM-DD for deterministic report generation.",
    )
    parser.add_argument(
        "--nhat-path", type=Path, default=Path("data/clean/polarity_nhat.csv")
    )
    parser.add_argument(
        "--long-path", type=Path, default=Path("data/clean/polarity_long.csv")
    )
    parser.add_argument(
        "--key-path", type=Path, default=Path("data/clean/polarity_key.csv")
    )
    parser.add_argument(
        "--disagreements-path",
        type=Path,
        default=Path("results/polarity/polarity_disagreements.csv"),
    )
    parser.add_argument(
        "--hash-path",
        type=Path,
        default=Path("results/polarity/polarity_files_hash.json"),
    )
    parser.add_argument(
        "--approval-path", type=Path, default=Path("docs/wtrans_approval.md")
    )
    parser.add_argument(
        "--prereg-path", type=Path, default=Path("docs/prereg.md")
    )
    parser.add_argument(
        "--out-wtrans-dir", type=Path, default=Path("data/wtrans")
    )
    parser.add_argument(
        "--out-rule-doc", type=Path, default=Path("docs/wtrans_rule.md")
    )
    parser.add_argument(
        "--constants-path", type=Path, default=Path("src/common/constants.py")
    )
    parser.add_argument("--n-per-emotion", type=int, default=50)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    run_build_wtrans_stage_b(
        date=args.date,
        nhat_path=args.nhat_path,
        long_path=args.long_path,
        key_path=args.key_path,
        disagreements_path=args.disagreements_path,
        hash_path=args.hash_path,
        approval_path=args.approval_path,
        prereg_path=args.prereg_path,
        out_wtrans_dir=args.out_wtrans_dir,
        out_rule_doc=args.out_rule_doc,
        constants_path=args.constants_path,
        expected_n_per_emotion=args.n_per_emotion,
        n_boot=args.n_boot,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
