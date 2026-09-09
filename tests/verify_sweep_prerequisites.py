"""Sweep ERP dependency, contrast fidelity, repair, and endpoint-status regressions."""
import asyncio
import copy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pipeline
import sweep

REQUEST = ("load sub-002.set, band-pass with high-pass swept over 0.1, 0.3 and 0.5 Hz "
           "and low-pass 30, average reference, epoch faces vs cars, "
           "measure the N170 mean over 110-150 ms at PO8")
SCOPE = {"codebook": {"conditions": {"faces": [[1, 40]], "cars": [[41, 80]]}}}
ENDPOINT = {"tool": "measure_component", "args": {
    "tmin": 0.11, "tmax": 0.15, "channels": ["PO8"], "mode": "mean"}}
BAD = {
    "base_pipeline": [
        {"tool": "load_eeg", "args": {"filepath": "/Users/fede/eeg-workspace/eeg-llm/data/erpcore_n170/sub-002/eeg/sub-002_task-N170_eeg.set"}},
        {"tool": "filter_eeg", "args": {"h_freq": 30, "method": "fir", "remove_dc": True}},
        {"tool": "set_reference", "args": {"ref_type": "average"}},
        {"tool": "create_bins", "args": {"bins": [
            {"label": "faces", "codes": list(range(1, 11))},
            {"label": "cars", "codes": list(range(101, 111))}], "require_following_code": 201}},
        {"tool": "create_epochs", "args": {"tmin": -0.2, "tmax": 0.5,
                                             "event_id": "faces", "baseline": [-0.2, 0]}},
        copy.deepcopy(ENDPOINT),
    ],
    "axes": [{"tool": "filter_eeg", "param": "l_freq", "values": [0.1, 0.3, 0.5]},
             {"tool": "create_epochs", "param": "event_id", "values": ["faces", "cars"]}],
    "endpoint": copy.deepcopy(ENDPOINT),
    "notes": ["6 variants (high-pass frequencies: 0.1, 0.3, 0.5 Hz; event_id: faces, cars)"],
}
GOOD = copy.deepcopy(BAD)
GOOD["axes"] = GOOD["axes"][:1]
GOOD["base_pipeline"][1]["args"] = {"h_freq": 30}
GOOD["base_pipeline"][3]["args"] = {"bins": [
    {"label": "faces", "codes": [[1, 40]]}, {"label": "cars", "codes": [[41, 80]]}]}
GOOD["base_pipeline"][4]["args"] = {"tmin": -0.2, "tmax": 0.5, "baseline": True}
GOOD["base_pipeline"][5] = {"tool": "compute_difference_erp", "args": {
    "event_id_a": "faces", "event_id_b": "cars", "tmin": -0.2, "tmax": 0.5}}
GOOD["notes"] = ["3 high-pass variants. Faces minus cars. Assumed epoch window -0.2 to 0.5 s and pre-stimulus baseline."]


class SweepTests(unittest.TestCase):
    def test_original_plan_is_rejected_without_running_or_resetting_session(self):
        self.assertIn("duplicates", " ".join(sweep.validate_sweep(BAD)))
        with patch.object(sweep, "_reset_session") as reset:
            run_step = AsyncMock()
            result = asyncio.run(sweep.run_sweep(BAD, run_step))
        self.assertFalse(result["ok"])
        reset.assert_not_called()
        run_step.assert_not_called()

    def test_missing_erp_is_rejected_even_when_endpoint_is_not_duplicated(self):
        spec = copy.deepcopy(GOOD)
        spec["base_pipeline"].pop()
        self.assertIn("no ERP producer", " ".join(sweep.validate_sweep(spec)))

    def test_erp_before_reloading_or_reepoching_does_not_satisfy_endpoint(self):
        for step in [GOOD["base_pipeline"][0], GOOD["base_pipeline"][4]]:
            spec = copy.deepcopy(GOOD)
            spec["base_pipeline"].append(step)
            self.assertIn("no ERP producer", " ".join(sweep.validate_sweep(spec)))

    def test_endpoint_schema_is_checked_before_execution(self):
        for endpoint in [{"tool": "unknown_tool", "args": {}},
                         {"tool": "measure_component", "args": {}},
                         {"tool": "measure_component", "args": {**ENDPOINT["args"], "tmin": "110"}}]:
            spec = copy.deepcopy(GOOD)
            spec["endpoint"] = endpoint
            self.assertTrue(sweep.validate_sweep(spec))
            with patch.object(sweep, "_reset_session") as reset:
                result = asyncio.run(sweep.run_sweep(spec, AsyncMock()))
                self.assertFalse(result["ok"])
                reset.assert_not_called()

    def test_clean_start_and_valid_three_variant_contrast(self):
        variants, errors = sweep.expand_sweep(GOOD)
        self.assertEqual(errors, [])
        self.assertEqual(len(variants), 3)
        self.assertEqual([v["assignments"]["filter_eeg.l_freq"] for v in variants], [0.1, 0.3, 0.5])
        self.assertEqual(sweep.validate_sweep_request(GOOD, REQUEST, SCOPE), [])
        spec = copy.deepcopy(GOOD)
        spec["base_pipeline"].pop(0)
        self.assertIn("empty session", " ".join(sweep.validate_sweep(spec)))

    def test_single_condition_erp_is_a_valid_producer(self):
        spec = copy.deepcopy(GOOD)
        spec["base_pipeline"][-1] = {"tool": "compute_erp", "args": {"tmin": -0.2, "tmax": 0.5}}
        self.assertEqual(sweep.validate_sweep(spec), [])
        self.assertTrue(sweep.validate_sweep_request(spec, REQUEST, SCOPE))

    def test_difference_requires_epoch_window_and_can_use_scoped_annotation_codes(self):
        spec = copy.deepcopy(GOOD)
        spec["base_pipeline"][-1]["args"].pop("tmax")
        self.assertIn("tmax", " ".join(sweep.validate_sweep(spec)))
        spec = copy.deepcopy(GOOD)
        spec["base_pipeline"][-1]["args"].update(
            event_id_a=[str(n) for n in range(1, 41)],
            event_id_b=[str(n) for n in range(41, 81)])
        self.assertEqual(sweep.validate_sweep_request(spec, REQUEST, SCOPE), [])
        spec["base_pipeline"][-1]["args"].update(event_id_a=["faces"], event_id_b=["cars"])
        self.assertEqual(sweep.validate_sweep_request(spec, REQUEST, SCOPE), [])
        spec["base_pipeline"][3]["args"]["bins"][0]["codes"] = [1, 40]
        self.assertTrue(sweep.validate_sweep_request(spec, REQUEST, SCOPE))

    def test_contrast_axes_and_scoped_event_codes_cannot_be_substituted(self):
        errors = sweep.validate_sweep_request(BAD, REQUEST, SCOPE)
        self.assertTrue(any("condition sweep axis" in e for e in errors))
        for change in ["codes", "order"]:
            spec = copy.deepcopy(GOOD)
            if change == "codes":
                spec["base_pipeline"][3]["args"]["bins"] = BAD["base_pipeline"][3]["args"]["bins"]
            else:
                spec["base_pipeline"][-1]["args"]["event_id_a"] = "cars"
            self.assertTrue(sweep.validate_sweep_request(spec, REQUEST, SCOPE))
        self.assertIn("codebook", " ".join(sweep.validate_sweep_request(GOOD, REQUEST, {})))

    def test_planner_repairs_original_response_and_rejects_exhausted_errors(self):
        for replies, valid, attempts in [([BAD, GOOD], True, 2), ([BAD] * 3, False, 3)]:
            with patch.object(pipeline.ollama, "chat", side_effect=[
                    {"message": {"content": json.dumps(r)}} for r in replies]):
                result = pipeline.propose_sweep("test", REQUEST, scope=SCOPE)
            self.assertEqual(result["_planner_meta"]["valid"], valid)
            self.assertEqual(result["_planner_meta"]["attempts"], attempts)

    def test_endpoint_failure_and_missing_or_nonfinite_amplitude_mark_row_failed(self):
        for endpoint_result in [{"ok": False, "error": "measurement failed"}, {"ok": True},
                                {"ok": True, "amplitude_uv": float("nan")}]:
            async def run_step(name, args):
                return (endpoint_result if name == "measure_component" else {"ok": True}), None
            with patch.object(sweep, "_reset_session"):
                result = asyncio.run(sweep.run_sweep(GOOD, run_step))
            self.assertEqual(len(result["rows"]), 3)
            self.assertTrue(all(not r["ok"] and r.get("error") for r in result["rows"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
