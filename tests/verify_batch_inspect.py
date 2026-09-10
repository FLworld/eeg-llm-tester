#!/usr/bin/env python3
"""Verify the /batch-inspect feature's mechanical pieces (no Ollama, no full batch):
  1. request detection + /batch disambiguation;
  2. the epochs save/reload roundtrip that lets a saved subject be re-opened with its bin labels
     intact (so a contrast can be re-averaged deterministically).
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FAILS = []
def check(c, m): (FAILS.append(m) or print(f"  FAIL: {m}")) if not c else print(f"  ok: {m}")

import app  # noqa: E402
t = app._batch_inspect_request
check(t("/batch-inspect sub-003") == "sub-003", "command with subject")
check(t("/batch-inspect") == "", "bare command -> usage")
check(t("/batch-inspect run1 sub-003") == "run1 sub-003", "command with run + subject")
check(t("inspect subject sub-005 from the batch") == "sub-005", "NL form")
check(t("inspect batch subject sub-002") == "sub-002", "NL 'batch subject' form")
check(t("/batch n170 data") is None, "plain /batch must NOT match /batch-inspect")
check(t("/name-batch pilot") is None, "/name-batch must NOT match /batch-inspect")
check(t("review the data") is None, "unrelated text does not match")

# /name-batch name validation (mirrors _handle_name_batch: reject spaces/slashes/'.'/'..')
import re as _re  # noqa: E402
_valid = lambda n: n not in (".", "..") and bool(_re.fullmatch(r"[A-Za-z0-9._-]+", n))
for n in ("pilot-v1", "n170_cohort", "runA.2"):
    check(_valid(n), f"valid batch name: {n!r}")
for n in ("..", ".", "has space", "a/b", "", "../x"):
    check(not _valid(n), f"rejected batch name: {n!r}")

# Epochs save -> reload roundtrip (guards _load_any epochs support + bin-label preservation)
import numpy as np  # noqa: E402
import mne  # noqa: E402
from tools import SESSION, save_eeg, load_eeg  # noqa: E402

info = mne.create_info(["PO8", "Oz"], 256.0, "eeg")
events = np.array([[0, 0, 1], [256, 0, 1], [512, 0, 2], [768, 0, 2]])
ep = mne.EpochsArray(np.random.randn(4, 2, 256) * 1e-6, info, events=events,
                     event_id={"B1": 1, "B2": 2}, tmin=-0.2, verbose="ERROR")
SESSION.clear()
SESSION.update({"raw": None, "epochs": ep, "ica": None, "bins": None, "filepath": "x"})
tmp = tempfile.mkdtemp()
res = save_eeg(os.path.join(tmp, "state.fif"))
check(res.get("ok") and res["saved"].endswith("_epo.fif"), "epochs save -> _epo.fif")

SESSION.clear()
SESSION.update({"raw": None, "epochs": None, "ica": None, "bins": None, "filepath": None})
lr = load_eeg(res["saved"])
check(lr.get("kind") == "epochs", "saved epochs reload as epochs (not raw)")
check(SESSION.get("epochs") is not None
      and {"B1", "B2"} <= set(SESSION["epochs"].event_id), "bin labels B1/B2 preserved on reload")

print("\nRESULT:", "PASS" if not FAILS else f"FAIL ({len(FAILS)})")
sys.exit(1 if FAILS else 0)
