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

# Guard: @cl.on_message must decorate on_message itself, not a helper defined just below it.
# (A helper inserted between the decorator and on_message silently breaks EVERY message.)
import re as _re2  # noqa: E402
import pathlib as _pl  # noqa: E402
_src = _pl.Path(app.__file__).read_text()
check(_re2.search(r"@cl\.on_message\s*\nasync def on_message\(", _src) is not None,
      "@cl.on_message decorates on_message (not an intervening helper)")

print("\nRESULT:", "PASS" if not FAILS else f"FAIL ({len(FAILS)})")
sys.exit(1 if FAILS else 0)
