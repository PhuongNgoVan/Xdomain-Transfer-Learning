"""Read-only, standard-library audit of every local branch and tracked data blob."""
import ast
from collections import Counter
import csv
import hashlib
import io
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT)


def bound_names(tree):
    names = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names.update(a.asname or a.name.split(".")[0] for a in node.names)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                names.update(n.id for n in ast.walk(target) if isinstance(n, ast.Name))
    return names


def main():
    branches = git("for-each-ref", "--format=%(refname:short)", "refs/heads").decode().splitlines()
    evidence = {"method": "AST + tracked Git blob checks; no pytest or model training", "branches": {}}
    for branch in branches:
        files = git("ls-tree", "-r", "--name-only", branch).decode("utf-8").splitlines()
        blobs = {f: git("show", f"{branch}:{f}") for f in files if f.endswith(".py")}
        trees = {}
        syntax_errors = []
        for f, raw in blobs.items():
            try:
                trees[f] = ast.parse(raw, filename=f"{branch}:{f}")
            except SyntaxError as exc:
                syntax_errors.append({"file": f, "line": exc.lineno, "error": str(exc)})
        missing_modules = []
        missing_symbols = []
        self_imports = []
        for f, tree in trees.items():
            for node in ast.walk(tree):
                if not isinstance(node, ast.ImportFrom) or not node.module or not node.module.startswith("src."):
                    continue
                target = node.module.replace(".", "/") + ".py"
                if target not in trees:
                    target = node.module.replace(".", "/") + "/__init__.py"
                loc = {"file": f, "line": node.lineno, "module": node.module}
                if target not in trees:
                    missing_modules.append(loc)
                    continue
                if target == f:
                    own_defs = {n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
                    self_imports.append({**loc, "symbols": [a.name for a in node.names], "locally_defined_functions": sorted(own_defs)})
                    continue
                known = bound_names(trees[target])
                for alias in node.names:
                    if alias.name not in known and alias.name != "*":
                        missing_symbols.append({**loc, "symbol": alias.name})
        manifests = {}
        for directory in ("data/clean", "data/splits"):
            manifest_file = f"{directory}/MANIFEST.json"
            if manifest_file not in files:
                continue
            manifest = json.loads(git("show", f"{branch}:{manifest_file}"))
            mismatch = []
            for name, expected in manifest.items():
                f = f"{directory}/{name}"
                if f not in files:
                    mismatch.append({"file": f, "error": "not tracked"})
                    continue
                actual = hashlib.sha256(git("show", f"{branch}:{f}")).hexdigest()
                if actual != expected:
                    mismatch.append({"file": f, "actual": actual, "expected": expected})
            manifests[directory] = {"entries": len(manifest), "mismatches": mismatch}
        requirements = git("show", f"{branch}:requirements.txt").decode()
        table = {
            "commit": git("rev-parse", branch).decode().strip(),
            "python_files": len(blobs), "syntax_errors": syntax_errors,
            "missing_local_modules": missing_modules,
            "missing_imported_symbols": missing_symbols,
            "self_imports": self_imports,
            "requirements": requirements.splitlines(),
            "manifests": manifests,
        }
        evidence["branches"][branch] = table
        print(json.dumps({"branch": branch, "python_files": len(blobs), "syntax_errors": len(syntax_errors),
                          "missing_modules": missing_modules, "missing_symbols": missing_symbols,
                          "self_imports": self_imports, "manifests": manifests}, ensure_ascii=True))

    # Independent integrity checks on the shared frozen CSVs, without pandas/torch.
    shared = {}
    for file in ("vsmec_train.csv", "vsmec_dev.csv", "vsmec_test.csv", "neuesc_100.csv", "vsfc_test_clean.csv"):
        blob = git("show", f"p1:data/clean/{file}")
        rows = list(csv.DictReader(io.StringIO(blob.decode("utf-8-sig"))))
        shared[file] = {"n": len(rows), "columns": list(rows[0]), "labels": dict(Counter(r["label"] for r in rows)),
                        "duplicate_ids": len(rows) - len({r["id"] for r in rows}),
                        "same_blob_all_branches": all(git("rev-parse", f"{b}:data/clean/{file}") == git("rev-parse", f"p1:data/clean/{file}") for b in branches)}
    evidence["shared_data"] = shared
    (OUT / "evidence.json").write_text(json.dumps(evidence, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"shared_data": shared}, ensure_ascii=True))


if __name__ == "__main__":
    main()
