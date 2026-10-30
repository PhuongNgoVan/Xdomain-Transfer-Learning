"""SHA-256 hashing utilities for files and text strings."""

import hashlib
from pathlib import Path
from typing import Dict, Iterable, Union

from src.common.textnorm import normalize_text

CHUNK_SIZE: int = 65536


def sha256_file(path: Union[str, Path]) -> str:
    """Compute the hexadecimal SHA-256 digest of a file.

    Args:
        path: Path to the file on disk.

    Raises:
        FileNotFoundError: If `path` does not exist or is not a file.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"File not found: {file_path}")

    hasher = hashlib.sha256()
    with file_path.open("rb") as f:
        for chunk in iter(lambda: f.read(CHUNK_SIZE), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def hash_text(text: str, normalize: bool = False) -> str:
    """Compute the hexadecimal SHA-256 digest of a UTF-8 text string.

    Args:
        text: Input string to hash.
        normalize: If True, apply `normalize_text` before hashing.

    Raises:
        TypeError: If `text` is not a `str`.
    """
    if not isinstance(text, str):
        raise TypeError(f"text must be a str, got {type(text).__name__}")

    target = normalize_text(text) if normalize else text
    return hashlib.sha256(target.encode("utf-8")).hexdigest()


def build_manifest(paths: Iterable[Union[str, Path]], base_dir: Union[str, Path, None] = None) -> Dict[str, str]:
    """Build a mapping from relative (or filename) paths to SHA-256 digests."""
    manifest: Dict[str, str] = {}
    base = Path(base_dir).resolve() if base_dir is not None else None
    for p in sorted(Path(x) for x in paths):
        resolved = p.resolve()
        key = resolved.relative_to(base).as_posix() if base is not None else p.name
        manifest[key] = sha256_file(resolved)
    return manifest
