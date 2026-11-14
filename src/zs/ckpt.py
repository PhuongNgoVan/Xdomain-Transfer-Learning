"""Find a unique baseline checkpoint and verify encoder and head integrity."""
import json
from pathlib import Path
import re

import torch

from src.common.hashing import sha256_file


def _run_pattern(alias, domain, seed):
    prefix = "Stage1-(?:base-)?(?:emokeep-)?bf16" if domain == "vsmec" else "NEU3-base(?:-emokeep)?-bf16"
    return re.compile(rf"^{re.escape(alias)}_{prefix}_s{int(seed)}_lr[0-9.eE+\-]+$")


def _weight_file(checkpoint):
    encoder = checkpoint / "encoder"
    config = encoder / "config.json"
    if not config.is_file():
        raise ValueError(f"Missing encoder config: {config}")
    try:
        metadata = json.loads(config.read_text(encoding="utf-8"))
    except (ValueError, OSError) as exc:
        raise ValueError(f"Invalid encoder config: {config}") from exc
    if not isinstance(metadata, dict):
        raise ValueError(f"Encoder config must be a JSON object: {config}")
    for filename in ("model.safetensors", "pytorch_model.bin"):
        weight = encoder / filename
        if weight.is_file() and weight.stat().st_size:
            return weight
    raise ValueError(f"Missing encoder model.safetensors/pytorch_model.bin: {encoder}")


def _verify_hashes(checkpoint, root, run_name, alias, domain, encoder_weight):
    filename = "stage1_ckpt_hash.json" if domain == "vsmec" else "neu3_ckpt_hash.json"
    directories = {root, root.parent, checkpoint, checkpoint.parent, checkpoint.parent.parent}
    manifests = set()
    for directory in directories:
        for path in (directory / filename, directory / "results" / filename):
            if path.is_file():
                manifests.add(path)
    if not manifests:
        return
    matches = []
    for manifest in sorted(manifests):
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or not payload:
            raise ValueError(f"Empty/invalid checkpoint hash manifest: {manifest}")
        for key, entry in payload.items():
            if not isinstance(entry, dict):
                continue
            recorded_parts = str(entry.get("path", "")).replace("\\", "/").split("/")
            if (key == run_name or run_name in recorded_parts
                    or (key == alias and run_name in manifest.parent.parts)):
                matches.append((manifest, entry))
            elif key == alias and Path(str(entry.get("path", ""))).resolve() == checkpoint.resolve():
                matches.append((manifest, entry))
    if not matches:
        raise ValueError(f"Hash manifest has no entry for {run_name}: {checkpoint}")
    for manifest, entry in matches:
        for key, path in (("encoder_sha256", encoder_weight), ("head_sha256", checkpoint / "head.pt")):
            if not entry.get(key) or sha256_file(path) != entry[key]:
                raise ValueError(f"Checkpoint {key} mismatch/missing in {manifest}: {path}")


def locate_checkpoint(root, alias, domain, seed):
    """Return one validated checkpoint directory; never load the encoder weights."""
    if domain not in ("vsmec", "neu3"):
        raise ValueError("domain must be vsmec or neu3")
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Checkpoint root does not exist: {root}")
    pattern = _run_pattern(alias, domain, seed)
    candidates = []
    for head in root.rglob("head.pt"):
        checkpoint = head.parent
        relatives = checkpoint.relative_to(root).parts
        if any("sweep" in p.lower() or "smoke" in p.lower() for p in (root.name, *relatives)):
            continue
        run_name = checkpoint.name if pattern.fullmatch(checkpoint.name) else checkpoint.parent.name
        if pattern.fullmatch(run_name) and (checkpoint.name == run_name or checkpoint.name == alias):
            candidates.append((checkpoint.resolve(), run_name))
    candidates = list(dict.fromkeys(candidates))
    if len(candidates) != 1:
        raise ValueError(f"Expected exactly one {alias}/{domain}/seed={seed} checkpoint, found {len(candidates)} under {root}")
    checkpoint, run_name = candidates[0]
    encoder_weight = _weight_file(checkpoint)
    head = torch.load(checkpoint / "head.pt", map_location="cpu", weights_only=True)
    weight, bias = head.get("weight"), head.get("bias")
    expected_labels = 7 if domain == "vsmec" else 3
    if not isinstance(weight, torch.Tensor) or weight.ndim != 2 or weight.shape[0] != expected_labels:
        raise ValueError(f"Invalid head width: expected {expected_labels} classes in {checkpoint}")
    if not isinstance(bias, torch.Tensor) or bias.shape != (expected_labels,):
        raise ValueError(f"Invalid head bias: {checkpoint}")
    _verify_hashes(checkpoint, root, run_name, alias, domain, encoder_weight)
    return checkpoint
