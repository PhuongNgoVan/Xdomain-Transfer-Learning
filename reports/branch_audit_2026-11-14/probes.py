"""Dependency-free probes of selected original AST nodes/functions.

These are focused reproductions, not pytest runs or end-to-end model tests.
The J3 probe substitutes only torch.load's head-shape result; it exercises the
original checkpoint-discovery and hash-verification control flow unchanged.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from types import ModuleType, SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent


def source(branch, file):
    return subprocess.check_output(["git", "show", f"{branch}:{file}"], cwd=ROOT).decode("utf-8")


def execute_nodes(nodes, namespace):
    tree = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *nodes], type_ignores=[])
    exec(compile(ast.fix_missing_locations(tree), "<selected-original-AST>", "exec"), namespace)


def main():
    findings = {}
    for branch in ("Neu3-phobert", "Neu3-visobert"):
        tree = ast.parse(source(branch, "src/plm/train_neu3.py"))
        node = next(n for n in tree.body if isinstance(n, ast.ImportFrom) and n.module == "src.plm.train_neu3")
        name = "src.plm.train_neu3"
        module = ModuleType(name)
        sys.modules[name] = module
        try:
            execute_nodes([node], module.__dict__)
        except ImportError as exc:
            findings[branch + ":self_import"] = {"confirmed": True, "error": str(exc), "line": node.lineno}
        else:
            raise AssertionError("Self import unexpectedly succeeded")
        finally:
            del sys.modules[name]

    tree = ast.parse(source("p1", "scripts/check_step2_data.py"))
    names = {"CheckResult", "_locate_group_seed_ckpt", "_find_prediction_csv_for_group", "check_group_j"}
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    const_names = {"GROUP_ORDER", "GROUP_LABELS", "GROUP_SPECS", "LEVEL_BLOCKER", "STATUS_PASS", "STATUS_FAIL", "STATUS_MISSING", "STATUS_SKIP"}
    constants = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            name = node.targets[0].id if isinstance(node.targets[0], ast.Name) else ""
        elif isinstance(node, ast.AnnAssign):
            name = node.target.id if isinstance(node.target, ast.Name) else ""
        else:
            continue
        if name in const_names:
            constants[name] = ast.literal_eval(node.value)
    ns = {"__name__": __name__, "Path": Path, "re": re, "json": json, "math": math,
          "dataclass": dataclass, "field": field,
          "sha256_file": lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest(), **constants}
    ns["torch"] = SimpleNamespace(load=lambda p, **kw: {"weight": SimpleNamespace(shape=(int(Path(p).read_text()), 4))})
    execute_nodes(nodes, ns)

    with tempfile.TemporaryDirectory(prefix="branch-audit-") as temp:
        root = Path(temp)
        ckpt_roots = {}
        hashes = {}
        for group, (_, alias, dim) in constants["GROUP_SPECS"].items():
            group_root = root / group
            ckpt_roots[group] = str(group_root)
            ckpt = group_root / "seed_0" / alias
            encoder = ckpt / "encoder"
            encoder.mkdir(parents=True)
            weights = encoder / "model.safetensors"
            weights.write_bytes(b"original encoder bytes")
            head = ckpt / "head.pt"
            head.write_text(str(dim))
            hashes[group] = {"path": str(ckpt), "head_sha256": ns["sha256_file"](head), "encoder_sha256": ns["sha256_file"](weights)}
        hash_path = root / "stage1_ckpt_hash.json"
        hash_path.write_text(json.dumps(hashes))
        cfg = {"ckpt_roots": ckpt_roots, "seeds": [0]}
        run = lambda: {r.id: r.status for r in ns["check_group_j"](cfg, root, check_ckpt=True)}
        before = run()
        for directory in ckpt_roots.values():
            for weights in Path(directory).rglob("model.safetensors"):
                weights.write_bytes(b"changed encoder bytes")
        after = run()
        assert before == after == {"J1": "PASS", "J2": "PASS", "J3": "PASS"}
        findings["p1:encoder_hash_not_verified"] = {"confirmed": True, "before": before, "after_encoder_changes": after,
                                                  "head_loader": "shape-only stub; real file bytes are hashed"}
        hash_path.write_text("{}")
        empty_hashes = run()
        assert empty_hashes["J3"] == "PASS"
        findings["p1:empty_hash_manifest_passes"] = {"confirmed": True, "checks": empty_hashes}

        preds = root / "predictions7"
        preds.mkdir()
        for seed in (0, 4):
            (preds / f"PhoBERT_Stage1-base_s{seed}_lr2e-5_vsmec_dev.csv").write_text("id\n1\n")
        selected = ns["_find_prediction_csv_for_group"]("vsmec_phobert", root / "seed_4" / "PhoBERT", [root])
        assert "_s0_" in selected.name
        findings["p1:prediction_seed_mismatch"] = {"confirmed": True, "checkpoint_seed": 4, "selected_csv": selected.name}

    # Execute the original test-record dictionary expression, with distinct dev/test metrics.
    rec_node = next(n for n in ast.walk(tree) if isinstance(n, ast.Dict) and any(isinstance(v, ast.Name) and v.id == "test_f1" for v in n.values))
    rec = eval(compile(ast.Expression(rec_node), "<original-test-record-expression>", "eval"), {"base_rec": {"f1_anger": 0.9}, "test_f1": 0.1})
    assert rec["split"] == "test" and rec["f1_anger"] == 0.9
    findings["p1:dev_class_f1_copied_to_test"] = {"confirmed": True, "record": rec, "line": rec_node.lineno}

    # Run the original result-file discovery function; no dataframe operations are needed.
    merge_tree = ast.parse(source("Neu3-phobert", "src/eval/merge_results.py"))
    excluded = next(n for n in merge_tree.body if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name) and n.target.id == "EXCLUDED_RESULT_FILENAMES")
    discover = next(n for n in merge_tree.body if isinstance(n, ast.FunctionDef) and n.name == "discover_result_files")
    merge_ns = {"Path": Path}
    execute_nodes([excluded, discover], merge_ns)
    with tempfile.TemporaryDirectory(prefix="branch-audit-merge-") as temp:
        (Path(temp) / "neu3.csv").write_text("run_id,backbone,stage1\nrun,PhoBERT,NEU\n")
        discovered = merge_ns["discover_result_files"](temp)
        assert [p.name for p in discovered] == ["neu3.csv"]
        findings["Neu3:merge_discovers_incompatible_neu3_csv"] = {"confirmed": True, "files": [p.name for p in discovered]}

    (OUT / "probes.json").write_text(json.dumps(findings, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(findings, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
