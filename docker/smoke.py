#!/usr/bin/env python3
"""Standalone runtime smoke test — baked into the image, independent of the dev tests/ tree.

Catches the failure mode Docker exists to prevent: a scientific-stack import breaking (esp. the
scipy<1.14 / sph_harm pin that silently disables the ICLabel path). Exits non-zero on any failure
so `make doctor` / CI can gate on it. Needs NO data and NO Ollama.
"""
import os
import sys

# App modules live in the working dir (/app in the image; the project dir locally). This script
# is invoked from elsewhere (/usr/local/bin), so put the app dir on the path explicitly.
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.environ.get("APP_DIR", "/app"))


def _check(label, fn):
    try:
        fn()
        print(f"  ✓ {label}")
        return True
    except Exception as exc:  # noqa: BLE001 - report, don't crash the runner
        print(f"  ✗ {label}: {exc.__class__.__name__}: {exc}")
        return False


def scipy_pin():
    import scipy
    from scipy.special import sph_harm  # noqa: F401 - the exact symbol ICLabel/autoreject need
    major, minor = (int(x) for x in scipy.__version__.split(".")[:2])
    assert (major, minor) < (1, 14), f"scipy {scipy.__version__} >= 1.14 (sph_harm removed)"


def imports():
    import mne  # noqa: F401
    import mne_icalabel  # noqa: F401
    import autoreject  # noqa: F401
    import chromadb  # noqa: F401
    import chainlit  # noqa: F401


def app_modules():
    import tools, pipeline, sweep, recipes, batch, qc, rag, exports  # noqa: F401


# Feature-conservation anchor: the shipped tool catalog must match the dev build exactly, so a
# feature can never silently drop demo->final. Bump this when you intentionally add/remove a tool.
EXPECTED_TOOL_COUNT = 34


def registry_parity():
    from tools import TOOL_FUNCTIONS, TOOL_SCHEMAS
    schema_names = {s["function"]["name"] for s in TOOL_SCHEMAS}
    fn_names = set(TOOL_FUNCTIONS)
    missing = schema_names ^ fn_names
    assert not missing, f"TOOL_FUNCTIONS/TOOL_SCHEMAS mismatch: {sorted(missing)}"
    assert len(fn_names) == EXPECTED_TOOL_COUNT, (
        f"tool count {len(fn_names)} != expected {EXPECTED_TOOL_COUNT} — a feature was dropped or "
        f"added (bump EXPECTED_TOOL_COUNT if intentional)")


def validate_trivial_pipeline():
    from pipeline import validate_pipeline
    cfg = {"pipeline": [
        {"tool": "load_eeg", "args": {"filepath": "sub-002.set"}},
        {"tool": "filter_eeg", "args": {"l_freq": 0.1, "h_freq": 30.0}},
    ]}
    errors = validate_pipeline(cfg)
    assert not errors, f"validate_pipeline rejected a valid plan: {errors}"


def main():
    print("eeg-llm runtime smoke:")
    results = [
        _check("scipy pin (<1.14, sph_harm importable)", scipy_pin),
        _check("science stack imports", imports),
        _check("app modules import", app_modules),
        _check("tool registry parity", registry_parity),
        _check("validate a trivial /plan", validate_trivial_pipeline),
    ]
    ok = all(results)
    print("\nSMOKE:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
