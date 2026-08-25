"""Hard provenance guards for resumable benchmark result directories."""

import hashlib
import json
from pathlib import Path


def file_fingerprint(path: str | Path) -> dict[str, int | str]:
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"sha256": digest.hexdigest(), "bytes": path.stat().st_size}


def ensure_config(out_dir: str | Path, filename: str, expected: dict,
                  protected_artifacts: tuple[str, ...]) -> Path:
    """Create a config once and reject unsafe resumptions thereafter."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename
    if path.exists():
        try:
            actual = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SystemExit(f"invalid result provenance file: {path}") from exc
        if actual != expected:
            raise SystemExit(
                f"result configuration mismatch in {path}; use a new --out directory"
            )
        return path

    existing = [name for name in protected_artifacts if (out_dir / name).exists()]
    if existing:
        raise SystemExit(
            f"cannot safely resume unverified artifacts in {out_dir}: "
            f"{', '.join(existing)}; use a new --out directory"
        )

    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(expected, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    return path


def require_config(out_dir: str | Path, filename: str,
                   required_artifacts: tuple[str, ...]) -> dict:
    """Require valid earlier-stage provenance and its result artifacts."""
    out_dir = Path(out_dir)
    path = out_dir / filename
    if not path.exists():
        raise SystemExit(
            f"missing result provenance file: {path}; rerun the earlier stage"
        )
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"invalid result provenance file: {path}") from exc
    if not isinstance(config, dict):
        raise SystemExit(f"invalid result provenance file: {path}")
    missing = [name for name in required_artifacts
               if not (out_dir / name).exists()]
    if missing:
        raise SystemExit(
            f"missing required result artifacts in {out_dir}: "
            f"{', '.join(missing)}; rerun the earlier stage"
        )
    return config
