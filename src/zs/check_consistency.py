"""Compare recomputed in-domain references against matching Step 1 test records."""
import argparse
from pathlib import Path
import pandas as pd

from src.common.schema import STAGE1_F1_COLUMNS


def check_consistency(results_dir, baseline_csv, baseline_pred_dir=None):
    results_dir = Path(results_dir)
    results = pd.read_csv(results_dir / "zs.csv", keep_default_na=False)
    baseline = pd.read_csv(baseline_csv, keep_default_na=False)
    if "split" not in baseline.columns:
        print("không có dòng test, bỏ qua")
        return pd.DataFrame()
    baseline = baseline.loc[baseline["split"].eq("test")].copy()
    if baseline.empty:
        print("không có dòng test, bỏ qua")
        return pd.DataFrame()
    if "stage1" in baseline.columns:
        domains = set(baseline["stage1"].astype(str).str.upper()) & {"VSMEC", "NEU"}
    elif set(STAGE1_F1_COLUMNS).issubset(baseline.columns):
        domains = {"VSMEC"}
    elif {"f1_neg", "f1_neu", "f1_pos"}.issubset(baseline.columns):
        domains = {"NEU"}
    else:
        raise ValueError("Cannot identify baseline domain from schema; do not compare unrelated references")
    if not domains:
        raise ValueError("No recognized baseline domain")
    reference = results.loc[results["mode"].eq("ref") & results["map_method"].eq("none")
                            & results["label_view"].isin(["V3-NEU", "V7-VSMEC"])
                            & results["stage1"].isin(domains)]
    records = []
    for _, row in reference.iterrows():
        alias = str(row["backbone"])
        candidates = baseline.loc[baseline["seed"].astype(int).eq(int(row["seed"]))]
        if "backbone" in candidates.columns:
            candidates = candidates.loc[candidates["backbone"].astype(str).str.contains(alias, case=False, regex=False)]
        if "stage1" in candidates.columns:
            candidates = candidates.loc[candidates["stage1"].astype(str).str.upper().eq(str(row["stage1"]).upper())]
        if "run_kind" in candidates.columns:
            candidates = candidates.loc[~candidates["run_kind"].eq("sweep")]
        if "run_id" in candidates.columns:
            candidates = candidates.loc[~candidates["run_id"].astype(str).str.contains("sweep|smoke", case=False, regex=True)]
        if candidates.empty:
            continue
        if len(candidates) != 1:
            raise ValueError(f"Ambiguous baseline records for {alias} seed {row['seed']}")
        baseline_row = candidates.iloc[0]
        diff = float(row["macro_f1"]) - float(baseline_row["macro_f1"])
        record = {"run_id": row["run_id"], "backbone": alias, "seed": int(row["seed"]),
                  "label_view": row["label_view"], "reference_f1": float(row["macro_f1"]),
                  "baseline_f1": float(baseline_row["macro_f1"]), "difference": diff,
                  "within_1e3": abs(diff) <= 1e-3}
        if baseline_pred_dir is not None and row["label_view"] == "V3-NEU":
            filename = f"{baseline_row['run_id']}_neuesc_test.csv"
            original = pd.read_csv(Path(baseline_pred_dir) / filename)
            recomputed = pd.read_csv(results_dir / "predictions_zs" / f"{row['run_id']}_neuesc_test.csv")
            if original["id"].duplicated().any() or recomputed["id"].duplicated().any():
                raise ValueError("Duplicate prediction IDs in consistency comparison")
            aligned = original[["id", "y_pred"]].merge(recomputed[["id", "y_pred3_sum"]], on="id", validate="one_to_one")
            if len(aligned) != len(original) or len(aligned) != len(recomputed):
                raise ValueError("Baseline and reference test populations differ")
            record["prediction_agreement"] = float(aligned["y_pred"].eq(aligned["y_pred3_sum"]).mean())
        if not record["within_1e3"]:
            print(f"WARN: {row['run_id']} baseline difference={diff:.6f}")
        records.append(record)
    table = pd.DataFrame(records)
    print(table.to_string(index=False) if not table.empty else "không có dòng test tương ứng, bỏ qua")
    return table


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--baseline-csv", required=True)
    parser.add_argument("--baseline-pred-dir")
    arguments = parser.parse_args(argv)
    check_consistency(arguments.results_dir, arguments.baseline_csv, arguments.baseline_pred_dir)


if __name__ == "__main__":
    main()
