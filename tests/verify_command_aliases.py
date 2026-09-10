#!/usr/bin/env python3
"""Verify slash-command separator normalization: users may type '_' or '-' in a command
(e.g. /lab_rules == /lab-rules), while argument underscores (filenames, JSON) stay intact."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import app  # noqa: E402

FAILS = []
def check(c, m): (FAILS.append(m) or print(f"  FAIL: {m}")) if not c else print(f"  ok: {m}")

n = app._normalize_slash_command
# command token normalized
check(n("/lab_rules") == "/lab-rules", "/lab_rules -> /lab-rules")
check(n("/inspect_ica 1 8") == "/inspect-ica 1 8", "/inspect_ica -> /inspect-ica")
check(n("/save_plan n170") == "/save-plan n170", "/save_plan -> /save-plan")
check(n("/batch_inspect demo_batch sub-002") == "/batch-inspect demo_batch sub-002",
      "/batch_inspect -> /batch-inspect")
# argument underscores preserved
check(n("/lab-rules demo_lab_rules.json") == "/lab-rules demo_lab_rules.json",
      "arg filename underscore preserved")
check(n("/scope sub_001.set") == "/scope sub_001.set", "arg recording underscore preserved")
check(n('/codebook {"a_b": 1}') == '/codebook {"a_b": 1}', "JSON arg underscore preserved")
# non-command text untouched
check(n("review the ica components") == "review the ica components", "NL text untouched")
check(n("") == "", "empty untouched")

print("\nRESULT:", "PASS" if not FAILS else f"FAIL ({len(FAILS)})")
sys.exit(1 if FAILS else 0)
