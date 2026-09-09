"""Create immutable, user-visible EEG analysis exports.

Exports deliberately live outside runtime checkpoints. A checkpoint is mutable working state;
an export is a shareable record of one completed pipeline or sweep and never overwrites another.
"""
from __future__ import annotations

import base64
import datetime
import json
import os
from pathlib import Path
import re
from typing import Callable


_HERE = Path(__file__).resolve().parent
OUTPUT_ROOT = Path(os.environ.get("EEG_OUTPUT_DIR", _HERE / "data-out"))


def slugify(name: str) -> str:
    """Make a user-provided analysis name safe for a single directory component."""
    return re.sub(r"[^a-z0-9]+", "-", (name or "").strip().lower()).strip("-")


def _unique_run_dir(root: Path, slug: str, timestamp: str) -> Path:
    base = root / "derivatives" / "eeg-llm"
    candidate = base / f"{slug}-{timestamp}"
    n = 2
    while candidate.exists():
        candidate = base / f"{slug}-{timestamp}-{n}"
        n += 1
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8")


def _write_figures(figures_dir: Path, images: list[tuple[str, str]] | None) -> list[str]:
    saved: list[str] = []
    for i, item in enumerate(images or [], 1):
        try:
            label, b64 = item
            data = base64.b64decode(b64, validate=True)
        except (TypeError, ValueError):
            continue
        stem = slugify(str(label)) or "figure"
        filename = f"{i:02d}-{stem}.png"
        (figures_dir / filename).write_bytes(data)
        saved.append(f"figures/{filename}")
    return saved


def _status(value) -> str:
    if isinstance(value, dict) and value.get("ok") is False:
        return "failed"
    return "ok"


def _report(name: str, kind: str, created: str, spec_name: str, results, scope: dict,
            data_file: str | None, figures: list[str]) -> str:
    lines = [f"# EEG-LLM export: {name}", "", f"Created: {created}", f"Run type: {kind}"]
    filepath = (scope or {}).get("filepath")
    if filepath:
        lines.append(f"Source recording: `{filepath}`")
    lines += ["", "## Contents", "", f"- `{spec_name}`: exact approved configuration"]
    if data_file:
        lines.append(f"- `{data_file}`: processed EEG snapshot")
    lines.append("- `scope.json`: recording structure, event inventory, and codebook")
    lines.append("- `results.json`: complete deterministic tool results")
    if figures:
        lines.append("- `figures/`: generated visual outputs")

    if kind == "pipeline":
        lines += ["", "## Executed Steps", "", "| # | Tool | Status |", "|---:|---|---|"]
        for i, row in enumerate(results or [], 1):
            lines.append(f"| {i} | `{row.get('tool', '?')}` | {_status(row.get('result'))} |")
    elif kind == "sweep":
        from sweep import format_sweep_table
        rows = (results or {}).get("rows", []) if isinstance(results, dict) else []
        lines += ["", "## Sweep Results", "", format_sweep_table(rows)]
        lines += ["", "A sweep does not save `processed.fif`: the in-memory EEG belongs only to "
                  "the final variant, not to the complete comparison."]
    return "\n".join(lines) + "\n"


def export_analysis(name: str, kind: str, spec: dict, results, scope: dict,
                    images: list[tuple[str, str]] | None = None,
                    save_snapshot: Callable[[str], dict] | None = None,
                    output_root: str | Path | None = None) -> dict:
    """Write one immutable export and return its user-visible relative location.

    ``save_snapshot`` is supplied by the application only for a completed single pipeline. It
    receives a target FIF stem and returns the same result shape as ``tools.save_eeg``.
    """
    slug = slugify(name)
    if not slug:
        return {"ok": False, "error": "An export needs a non-empty analysis name."}
    if kind not in {"pipeline", "sweep"}:
        return {"ok": False, "error": f"Unsupported export kind {kind!r}."}

    created = datetime.datetime.now().isoformat(timespec="seconds")
    timestamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    try:
        root = Path(output_root) if output_root is not None else OUTPUT_ROOT
        run_dir = _unique_run_dir(root, slug, timestamp)
        figures_dir = run_dir / "figures"
        figures_dir.mkdir()

        data_result, data_file = None, None
        if save_snapshot is not None:
            data_result = save_snapshot(str(run_dir / "processed.fif"))
            if not isinstance(data_result, dict) or not data_result.get("ok"):
                return {"ok": False, "error": (data_result or {}).get(
                    "error", "Could not save the processed EEG snapshot."), "path": str(run_dir)}
            data_file = Path(data_result["saved"]).name

        spec_name = "pipeline.json" if kind == "pipeline" else "sweep.json"
        _write_json(run_dir / spec_name, spec)
        _write_json(run_dir / "scope.json", scope or {})
        _write_json(run_dir / "results.json", results)
        figures = _write_figures(figures_dir, images)
        (run_dir / "report.md").write_text(
            _report(name, kind, created, spec_name, results, scope or {}, data_file, figures),
            encoding="utf-8")
        manifest = {
            "name": name,
            "slug": slug,
            "created": created,
            "kind": kind,
            "source_recording": (scope or {}).get("filepath"),
            "configuration": spec_name,
            "processed_data": data_file,
            "figures": figures,
        }
        _write_json(run_dir / "manifest.json", manifest)
    except Exception as exc:  # noqa: BLE001 - surface a filesystem failure to the chat UI
        return {"ok": False, "error": f"Export failed: {exc}"}

    relative = run_dir.relative_to(root)
    return {"ok": True, "path": str(run_dir), "relative_path": str(relative),
            "data_file": data_file, "figures": figures}
