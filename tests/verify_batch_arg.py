#!/usr/bin/env python3
"""Verify /batch argument parsing is robust to spaces and quotes in the dataset dir.

Regression for the tester bug: a cohort dir named 'erpcore_n170 copy' made `arg.split()` treat
'copy' as the dataset dir and 'erpcore_n170' as a (nonexistent) recipe name, so the batch never
ran. The parser must disambiguate by which candidate actually exists (as typed or under DATA_DIR),
never by whitespace.
"""
import os
import sys
import tempfile

# Point DATA_DIR at a temp tree with a spaced cohort dir BEFORE importing app.
_TMP = tempfile.mkdtemp()
os.makedirs(os.path.join(_TMP, "erpcore_n170 copy", "sub-001", "eeg"), exist_ok=True)
os.makedirs(os.path.join(_TMP, "erpcore_n170", "sub-001", "eeg"), exist_ok=True)
os.environ["EEG_DATA_DIR"] = _TMP

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import app  # noqa: E402

FAILS = []


def check(cond, msg):
    (FAILS.append(msg) or print(f"  FAIL: {msg}")) if not cond else print(f"  ok: {msg}")


spaced = os.path.join(_TMP, "erpcore_n170 copy")
plain = os.path.join(_TMP, "erpcore_n170")

# bare spaced dir -> no recipe, resolved
r, raw, res = app._split_batch_arg("erpcore_n170 copy")
check(r is None and res == spaced, "bare spaced dir resolves, no recipe")

# quoted spaced dir -> no recipe, resolved
r, raw, res = app._split_batch_arg('"erpcore_n170 copy"')
check(r is None and res == spaced, "quoted spaced dir resolves")

# recipe + spaced dir
r, raw, res = app._split_batch_arg("n170 erpcore_n170 copy")
check(r == "n170" and res == spaced, "recipe + spaced dir")

# recipe + quoted spaced dir
r, raw, res = app._split_batch_arg('n170 "erpcore_n170 copy"')
check(r == "n170" and res == spaced, "recipe + quoted spaced dir")

# plain dir still works, no recipe
r, raw, res = app._split_batch_arg("erpcore_n170")
check(r is None and res == plain, "plain dir, no recipe")

# recipe + plain dir
r, raw, res = app._split_batch_arg("n170 erpcore_n170")
check(r == "n170" and res == plain, "recipe + plain dir")

# nonexistent -> resolved None (handler reports)
r, raw, res = app._split_batch_arg("does_not_exist")
check(res is None, "nonexistent dir -> None (reported to user)")

# absolute path passthrough
r, raw, res = app._split_batch_arg(spaced)
check(res == spaced, "absolute spaced path resolves")

print("\nRESULT:", "PASS" if not FAILS else f"FAIL ({len(FAILS)})")
sys.exit(1 if FAILS else 0)
