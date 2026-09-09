#!/usr/bin/env python3
"""Verify the deterministic 'remove ICA components -> [epoch A vs B] -> [measure]' executor's
parsing + routing. No Ollama, no data: exercises the pure functions in app.py.

Guards the invariants that make it faithful:
  - the epoch window is NEVER inferred from the measurement window (must be stated separately);
  - a numberless "review ... components" is NOT captured as an apply request;
  - condition names resolve tolerantly (plural/case) to codebook codes.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import app  # noqa: E402

FAILS = []


def check(cond, msg):
    if not cond:
        FAILS.append(msg)
        print(f"  FAIL: {msg}")
    else:
        print(f"  ok: {msg}")


# --- detector: only "remove/drop/reject/exclude ... ICA components <n>" ---
for t in ["remove ICA components 1 and 7", "drop ica component 3",
          "Reject ICA components 2, 4 and 6"]:
    check(app._apply_ica_compound_request(t), f"detects: {t!r}")
for t in ["review the ica components", "review the ica components 1 and 8",
          "run ICA", "inspect components 1 and 8", "epoch faces vs cars"]:
    check(not app._apply_ica_compound_request(t), f"does NOT capture: {t!r}")

# --- full line with an explicit epoch window ---
p = app._parse_apply_ica_compound(
    "Remove ICA components 1 and 7, then epoch faces vs cars from -200 to 800 ms "
    "and measure the mean 110-150 ms at PO8")
check(p["components"] == [1, 7], "components [1,7]")
check((p["cond_a"], p["cond_b"]) == ("faces", "cars"), "conditions faces/cars")
check(p.get("epoch_tmin") == -0.2 and p.get("epoch_tmax") == 0.8, "epoch window -0.2..0.8 s")
check(p.get("meas_tmin") == 0.11 and p.get("meas_tmax") == 0.15, "measure window 0.11..0.15 s")
check(p["channel"] == "PO8", "channel PO8")
check(p["mode"] == "mean" and not p["errors"], "mode mean, no errors")

# --- CRITICAL: no epoch window given -> epoch window must be ABSENT (never the measure window) ---
p2 = app._parse_apply_ica_compound(
    "Remove ICA components 1 and 7, then epoch faces vs cars and measure the mean 110-150 ms at PO8")
check("epoch_tmin" not in p2, "epoch window is NOT inferred from the measurement window")
check(p2.get("meas_tmin") == 0.11 and p2.get("meas_tmax") == 0.15, "measurement window still parsed")

# --- peak + alternate channel + 'to' separator + seconds unit ---
p3 = app._parse_apply_ica_compound(
    "drop ica components 3 and 4, then epoch face vs car from -0.2 to 0.8 s "
    "and measure the peak 130 to 200 ms at PO7")
check(p3.get("epoch_tmin") == -0.2 and p3.get("epoch_tmax") == 0.8, "seconds unit epoch window")
check(p3.get("meas_tmin") == 0.13 and p3.get("meas_tmax") == 0.2, "peak window 0.13..0.2 s")
check(p3["mode"] == "peak" and p3["channel"] == "PO7", "mode peak, channel PO7")

# --- remove-only (no analysis) ---
p4 = app._parse_apply_ica_compound("remove ICA components 1 and 7")
check(p4["components"] == [1, 7] and not p4["wants_epoch"] and not p4["wants_measure"],
      "remove-only: components only, no epoch/measure")

# --- condition resolution tolerant of plural/case ---
cond = {"face": [[1, 40]], "car": [[41, 80]]}
check(app._resolve_condition_codes("faces", cond) == [[1, 40]], "'faces' -> face codes")
check(app._resolve_condition_codes("CARS", cond) == [[41, 80]], "'CARS' -> car codes")
check(app._resolve_condition_codes("houses", cond) is None, "unknown condition -> None")

print("\nRESULT:", "PASS" if not FAILS else f"FAIL ({len(FAILS)})")
sys.exit(1 if FAILS else 0)
