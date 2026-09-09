"""Isolated browser fixture with a real ICA fitted to explicitly synthetic data."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from verify_inspect_ica import synthetic_ica
import app

raw, ica = synthetic_ica()
app.SESSION.update(raw=raw, ica=ica, epochs=None)
