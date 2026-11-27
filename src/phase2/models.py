"""Fresh target heads with verified source encoders for DFT, Seq and C0."""
import json
from pathlib import Path

from transformers import AutoModel, AutoTokenizer
import yaml

from src.common.hashing import sha256_file
from src.common.seed import set_seed
from src.plm.model_ckpt import Stage1Classifier, _find_encoder_weight_file
from src.zs.ckpt import locate_checkpoint


def count_model_parameters(model):
    return {"total_params": sum(p.numel() for p in model.parameters()),
            "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad)}


def file_hashes(root):
    root = Path(root).resolve()
    return {p.relative_to(root).as_posix(): sha256_file(p)
            for p in sorted(root.rglob("*")) if p.is_file()}


def artifact_path(run_dir, relative):
    """Completion paths are relative to their run directory (including checkpoint refs)."""
    return (Path(run_dir) / relative).resolve()


def verify_c0_stage1_dft_checkpoint(dft_run_dir, expected_direction, expected_fraction,
                                    expected_seed, expected_ids_sha256, protocol_cfg):
    root = Path(dft_run_dir).resolve()
    completion = json.loads((root / "completion.json").read_text(encoding="utf-8"))
    meta = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    def reject(reason):
        raise ValueError(f"C0 low-resource violation: {reason}")
    if completion.get("complete") is not True:
        reject("DFT run is incomplete")
    for key, expected in (("protocol_version", protocol_cfg["protocol_version"]),
                          ("method", "dft"), ("direction", expected_direction),
                          ("fraction", int(expected_fraction)), ("seed", int(expected_seed)),
                          ("run_mode", "main"), ("max_examples", None)):
        if meta.get(key) != expected:
            reject(f"stage1 {key} mismatch")
    if not expected_ids_sha256 or meta["data_provenance"]["ids_sha256"] != expected_ids_sha256:
        reject("stage1 ids_sha256 mismatch")
    if meta["data_provenance"].get("max_examples") is not None:
        reject("truncated DFT run")
    if not completion.get("ckpt_path"):
        reject("missing checkpoint")
    ckpt = artifact_path(root, completion["ckpt_path"])
    artifacts = completion.get("artifacts", {})
    required = [root / "metadata.json", root / "config.yaml", root / "cost.json",
                ckpt / "head.pt", _find_encoder_weight_file(ckpt / "encoder")]
    required.extend(p for p in (ckpt / "encoder").rglob("*") if p.is_file())
    recorded = {artifact_path(root, name): digest for name, digest in artifacts.items()}
    if any(p.resolve() not in recorded for p in required):
        reject("checkpoint/config/metadata/cost not registered in completion")
    if any(not p.is_file() or sha256_file(p) != digest for p, digest in recorded.items()):
        reject("artifact hash mismatch")
    cfg = yaml.safe_load((root / "config.yaml").read_text(encoding="utf-8"))
    if cfg["protocol_cfg"] != dict(protocol_cfg):
        reject("protocol config mismatch")
    return {"stage1_dft_run_id": meta["run_id"], "stage1_dft_ckpt_path": str(ckpt),
            "stage1_ids_sha256": expected_ids_sha256,
            "stage1_encoder_sha256": sha256_file(_find_encoder_weight_file(ckpt / "encoder")),
            "stage1_cost_ref": str(root / "cost.json"),
            "stage1_cost_sha256": sha256_file(root / "cost.json"),
            "stage1_source_files_sha256": meta["data_provenance"]["source_files_sha256"],
            "stage1_backbone": meta["backbone"], "stage1_fingerprint": completion["fingerprint"]}


def _verify_xlmr_run_manifest(ckpt, source_dir, seed, encoder_weight, results_dir=None, protocol_cfg=None):
    run_id = ckpt.name
    search_dirs = []
    if results_dir is not None:
        res_root = Path(results_dir).resolve()
        res_runs = res_root / "runs"
        expected_run_dir = res_runs / run_id
        if res_runs.is_dir() and any(res_runs.glob("XLMR_*")) and not expected_run_dir.is_dir():
            raise ValueError(f"Invalid XLMR Stage-1 source checkpoint: missing run directory {expected_run_dir}")
        search_dirs.append(expected_run_dir)
    search_dirs.extend([
        ckpt.parent.parent / "results" / "phase2" / "runs" / run_id,
        ckpt.parent.parent.parent / "results" / "phase2" / "runs" / run_id,
        ckpt.parent / "runs" / run_id,
    ])
    run_dir = None
    for candidate in search_dirs:
        if candidate.is_dir() and ((candidate / "completion.json").exists() or (candidate / "metadata.json").exists()):
            run_dir = candidate
            break
    if run_dir is None:
        return {"run_id": run_id, "verified_completion": False}

    comp_path = run_dir / "completion.json"
    meta_path = run_dir / "metadata.json"
    if not comp_path.is_file() or not meta_path.is_file():
        raise ValueError(f"Invalid XLMR Stage-1 source checkpoint: missing completion/metadata in {run_dir}")
    completion = json.loads(comp_path.read_text(encoding="utf-8"))
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if completion.get("complete") is not True:
        raise ValueError(f"Invalid XLMR Stage-1 source checkpoint: incomplete run {run_id}")
    if (
        meta.get("backbone") != "XLMR"
        or meta.get("method") != "dft"
        or meta.get("direction") != source_dir
        or int(meta.get("fraction", -1)) != 100
        or int(meta.get("seed", -1)) != int(seed)
        or meta.get("run_mode") != "main"
        or meta.get("max_examples") is not None
        or (meta.get("data_provenance") or {}).get("max_examples") is not None
    ):
        raise ValueError(f"Invalid XLMR Stage-1 source checkpoint: metadata mismatch or truncated run {run_id}")
    if protocol_cfg is not None and meta.get("protocol_version") != protocol_cfg["protocol_version"]:
        raise ValueError(f"Invalid XLMR Stage-1 source checkpoint: protocol_version mismatch for {run_id}")

    artifacts = completion.get("artifacts", {})
    for req_name in ("metadata.json", "config.yaml"):
        req_file = run_dir / req_name
        if req_name not in artifacts or not req_file.is_file() or sha256_file(req_file) != artifacts[req_name]:
            raise ValueError(f"Invalid XLMR Stage-1 source checkpoint: artifact hash mismatch for {run_id}/{req_name}")

    head_entries = [digest for rel, digest in artifacts.items() if rel == "head.pt" or rel.endswith("/head.pt")]
    enc_entries = [digest for rel, digest in artifacts.items() if rel.endswith(f"/encoder/{encoder_weight.name}")]
    if not head_entries or sha256_file(ckpt / "head.pt") != head_entries[0]:
        raise ValueError(f"Invalid XLMR Stage-1 source checkpoint: head.pt hash mismatch for {run_id}")
    if not enc_entries or sha256_file(encoder_weight) != enc_entries[0]:
        raise ValueError(f"Invalid XLMR Stage-1 source checkpoint: encoder hash mismatch for {run_id}")
    return {"run_id": meta["run_id"], "verified_completion": True}


def _locate_phase2_xlmr_source_checkpoint(root, direction, seed, *, results_dir=None, protocol_cfg=None, return_meta=False):
    """Locate and verify a unique Phase 2 XLMR DFT 100% source checkpoint for Seq/ZS."""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Checkpoint root does not exist: {root}")
    source_dir = "SB" if direction == "SA" else "SA"
    expected_labels = 7 if direction == "SA" else 3
    prefix = f"XLMR_{source_dir}_dft_f100_s{int(seed)}_"
    candidates = [
        p.resolve() for p in root.rglob(f"{prefix}*")
        if p.is_dir() and (p / "head.pt").is_file() and (p / "encoder").is_dir()
        and not any("sweep" in part.lower() or "smoke" in part.lower()
                    for part in (root.name, *p.relative_to(root).parts))
    ]
    candidates = list(dict.fromkeys(candidates))
    if len(candidates) != 1:
        raise ValueError(
            f"Expected exactly one XLMR/{direction}/seed={seed} source checkpoint, "
            f"found {len(candidates)} under {root}"
        )
    ckpt = candidates[0]
    encoder_weight = _find_encoder_weight_file(ckpt / "encoder")
    import torch
    head = torch.load(ckpt / "head.pt", map_location="cpu", weights_only=True)
    weight, bias = head.get("weight"), head.get("bias")
    if (
        not isinstance(weight, torch.Tensor)
        or weight.ndim != 2
        or weight.shape[0] != expected_labels
        or not isinstance(bias, torch.Tensor)
        or bias.shape != (expected_labels,)
    ):
        raise ValueError(f"Invalid head width/bias: expected {expected_labels} classes in {ckpt}")
    meta_info = _verify_xlmr_run_manifest(
        ckpt, source_dir, seed, encoder_weight, results_dir=results_dir, protocol_cfg=protocol_cfg
    )
    return (ckpt, meta_info) if return_meta else ckpt


def resolve_model_source(method, direction, backbone, seed, fraction, protocol_cfg, *,
                         pretrained_model_path=None, source_ckpt_root=None,
                         c0_stage1_run_dir=None, expected_ids_sha256=None, results_dir=None):
    if method == "dft":
        ref = str(pretrained_model_path or protocol_cfg["backbones"][backbone]["model_name"])
        return ref, {"init_kind": "pretrained_backbone", "model_ref": ref,
                     "local_files_sha256": file_hashes(ref) if Path(ref).is_dir() else {}}
    if method == "seq":
        if source_ckpt_root is None:
            raise ValueError("Seq requires source_ckpt_root")
        domain = "vsmec" if direction == "SA" else "neu3"
        if backbone == "XLMR":
            ckpt, xlmr_meta = _locate_phase2_xlmr_source_checkpoint(
                source_ckpt_root, direction, seed,
                results_dir=results_dir, protocol_cfg=protocol_cfg, return_meta=True,
            )
            return str(ckpt / "encoder"), {
                "init_kind": "source_checkpoint",
                "src_ckpt_path": str(ckpt),
                "src_domain": domain,
                "encoder_sha256": sha256_file(_find_encoder_weight_file(ckpt / "encoder")),
                "source_files_sha256": file_hashes(ckpt),
                "source_run_origin": "phase2_xlmr_dft_100pct",
                "source_dft_run_id": xlmr_meta["run_id"],
                "source_dft_verified": xlmr_meta["verified_completion"],
            }
        ckpt = locate_checkpoint(source_ckpt_root, backbone, domain, seed)
        return str(ckpt / "encoder"), {"init_kind": "source_checkpoint", "src_ckpt_path": str(ckpt),
                "src_domain": domain, "encoder_sha256": sha256_file(_find_encoder_weight_file(ckpt / "encoder")),
                "source_files_sha256": file_hashes(ckpt), "historical_run_provenance": "unverified"}
    if method == "c0":
        if c0_stage1_run_dir is None:
            raise ValueError("C0 requires c0_stage1_run_dir")
        provenance = verify_c0_stage1_dft_checkpoint(c0_stage1_run_dir, direction, fraction,
                                                    seed, expected_ids_sha256, protocol_cfg)
        if provenance["stage1_backbone"] != backbone:
            raise ValueError("C0 low-resource violation: backbone mismatch")
        return str(Path(provenance["stage1_dft_ckpt_path"]) / "encoder"), provenance
    raise ValueError(f"Unsupported method: {method}")


def build_phase2_model(method, direction, backbone, seed, fraction, protocol_cfg, *,
                       pretrained_model_path=None, source_ckpt_root=None, c0_stage1_run_dir=None,
                       expected_ids_sha256=None, results_dir=None, local_files_only=False, model_adapter_fn=None):
    ref, provenance = resolve_model_source(method, direction, backbone, seed, fraction, protocol_cfg,
        pretrained_model_path=pretrained_model_path, source_ckpt_root=source_ckpt_root,
        c0_stage1_run_dir=c0_stage1_run_dir, expected_ids_sha256=expected_ids_sha256, results_dir=results_dir)
    local = local_files_only or method != "dft"
    encoder = AutoModel.from_pretrained(ref, local_files_only=local)
    tokenizer = AutoTokenizer.from_pretrained(ref, local_files_only=local)
    set_seed(int(seed))
    model = Stage1Classifier(encoder, protocol_cfg["directions"][direction]["target_num_labels"],
                             dropout=protocol_cfg["training"]["dropout"])
    if model_adapter_fn is not None:
        model = model_adapter_fn(model, protocol_cfg)
    return model, tokenizer, {**provenance, "head_seed": int(seed), **count_model_parameters(model)}
