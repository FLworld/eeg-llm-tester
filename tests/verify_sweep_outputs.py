"""Non-ERP sweeps preserve their final results, figures, failures, and exports."""
import asyncio
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mne
import numpy as np
import pipeline
import sweep
import tools
from exports import export_analysis


def filter_spec(path):
    return {"base_pipeline": [
        {"tool": "load_eeg", "args": {"filepath": str(path)}},
        {"tool": "filter_eeg", "args": {"l_freq": 1, "h_freq": 30, "method": "iir", "iir_order": 2}}],
        "axes": [{"tool": "filter_eeg", "param": "h_freq", "values": [20, 30]}]}


async def real_step(name, args):
    result = tools.call_tool(name, args)
    return result, result.pop("image", None)


class OutputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.path = Path(cls.temp.name) / "synthetic-raw.fif"
        rng = np.random.default_rng(17)
        names = ["Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4"]
        raw = mne.io.RawArray(rng.normal(size=(8, 8)) @ rng.laplace(size=(8, 2560)) * 1e-6,
                             mne.create_info(names, 128, "eeg"), verbose="ERROR")
        raw.set_montage("standard_1020")
        raw.save(cls.path, overwrite=True, verbose="ERROR")

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def run_spec(self, spec):
        with patch.dict(tools.SESSION, {}, clear=True):
            result = asyncio.run(sweep.run_sweep(spec, real_step))
        self.assertTrue(result["ok"], result)
        self.assertTrue(all(r["ok"] for r in result["rows"]), result["rows"])
        return result

    def test_filter_only_has_no_erp_requirement_or_amplitude_column(self):
        result = self.run_spec(filter_spec(self.path))
        self.assertEqual([r["output"]["h_freq"] for r in result["rows"]], [20, 30])
        self.assertTrue(all(r["output_tool"] == "filter_eeg" and "endpoint_uv" not in r for r in result["rows"]))
        table = sweep.format_sweep_table(result["rows"])
        self.assertIn("h_freq", table)
        self.assertNotIn("uV", table)
        self.assertTrue(all(len(r["steps"]) == 2 for r in result["rows"]))
        self.assertEqual(len(result["images"]), 3)
        self.assertEqual(result["comparison_figure"], "filter-psd-comparison")
        spectra = [r["diagnostics"]["psd"]["result"] for r in result["rows"]]
        self.assertTrue(all(p["fmin"] == 0 and p["fmax"] == 64 for p in spectra))
        freqs = np.asarray(spectra[0]["freqs_hz"])
        self.assertEqual(spectra[0]["freqs_hz"], spectra[1]["freqs_hz"])
        band = (freqs >= 25) & (freqs <= 35)
        self.assertLess(np.asarray(spectra[0]["mean_psd_v2_hz"])[band].sum(),
                        np.asarray(spectra[1]["mean_psd_v2_hz"])[band].sum())
        self.assertTrue(all(p["psd_units"] == "V^2/Hz" for p in spectra))
        with patch.object(tools, "_fig_to_b64", side_effect=lambda fig: self.capture_comparison(fig)):
            sweep._plot_filter_psd_comparison(result["rows"])
        exported = export_analysis("Filter PSD", "sweep", filter_spec(self.path), result, {},
                                   images=result["images"], output_root=self.temp.name)
        folder = Path(exported["path"])
        self.assertEqual(len(list((folder / "figures").glob("*.png"))), 3)
        saved = json.loads((folder / "results.json").read_text())
        self.assertEqual(saved["rows"][0]["diagnostics"]["psd"]["result"], spectra[0])
        self.assertIn("read-only PSD diagnostic", (folder / "report.md").read_text())

    def capture_comparison(self, fig):
        import matplotlib.pyplot as plt
        self.assertEqual(len(fig.axes[0].lines), 2)
        self.assertIn("PSD", fig.axes[0].get_ylabel())
        self.assertEqual(fig.axes[0].lines[0].get_label(), "filter_eeg.h_freq=20")
        plt.close(fig)
        return "checked"

    def test_psd_units_match_mne_and_diagnostic_is_read_only(self):
        with patch.dict(tools.SESSION, {}, clear=True):
            tools.load_eeg(str(self.path))
            raw = tools.SESSION["raw"]
            before = raw.get_data().copy()
            expected, freqs = raw.compute_psd(fmin=0, fmax=64, picks="eeg").get_data(return_freqs=True)
            def capture(fig):
                import matplotlib.pyplot as plt
                np.testing.assert_allclose(fig.axes[0].lines[0].get_ydata(), expected.mean(axis=0) * 1e12)
                plt.close(fig)
                return "image"
            with patch.object(tools, "_fig_to_b64", side_effect=capture):
                result = tools.compute_psd(fmin=0, fmax=None)
            np.testing.assert_allclose(result["mean_psd_v2_hz"], expected.mean(axis=0))
            np.testing.assert_array_equal(before, raw.get_data())

    def test_psd_diagnostic_failure_preserves_filter_result_and_is_visible(self):
        async def failing_psd(name, args):
            if name == "compute_psd":
                raise RuntimeError("PSD unavailable")
            return await real_step(name, args)
        with patch.dict(tools.SESSION, {}, clear=True):
            result = asyncio.run(sweep.run_sweep(filter_spec(self.path), failing_psd))
        self.assertTrue(all(row["ok"] for row in result["rows"]))
        self.assertTrue(all(not row["diagnostics"]["psd"]["result"]["ok"] for row in result["rows"]))
        self.assertIn("failed: PSD unavailable", sweep.format_sweep_table(result["rows"]))
        self.assertNotIn("comparison_figure", result)

    def test_ica_algorithm_outputs_and_figures_are_preserved_and_exportable(self):
        spec = filter_spec(self.path)
        spec["base_pipeline"].append({"tool": "run_ica", "args": {
            "algorithm": "infomax", "n_components": 4, "random_state": 42,
            "label_components": False, "iclabel_preprocessing": False, "max_iter": 100}})
        spec["axes"] = [{"tool": "run_ica", "param": "algorithm", "values": ["infomax", "fastica"]}]
        result = self.run_spec(spec)
        self.assertEqual(len(result["images"]), 2)
        self.assertTrue(all("diagnostics" not in row for row in result["rows"]))
        self.assertEqual([r["output"]["n_components"] for r in result["rows"]], [4, 4])
        self.assertEqual(len({r["output"]["method"] for r in result["rows"]}), 2)
        self.assertNotIn("uV", sweep.format_sweep_table(result["rows"]))
        self.assertTrue(all("endpoint_uv" not in r for r in result["rows"]))
        images = result.pop("images")
        exported = export_analysis("ICA comparison", "sweep", spec, result, {}, images=images,
                                   output_root=self.temp.name)
        self.assertTrue(exported["ok"], exported)
        folder = Path(exported["path"])
        self.assertEqual(len(list((folder / "figures").glob("*.png"))), 2)
        self.assertIn("run_ica", (folder / "report.md").read_text())
        self.assertNotIn("uV", (folder / "report.md").read_text())
        self.assertEqual(json.loads((folder / "results.json").read_text())["rows"][0]["output_tool"], "run_ica")

    def test_requested_psd_endpoint_is_not_replaced_with_erp(self):
        spec = filter_spec(self.path)
        spec["endpoint"] = {"tool": "compute_psd", "args": {"fmin": 1, "fmax": 40}}
        result = self.run_spec(spec)
        self.assertEqual(len(result["images"]), 2)
        self.assertTrue(all(r["output_tool"] == "compute_psd" for r in result["rows"]))
        self.assertTrue(all("diagnostics" not in row and len(row["steps"]) == 3 for row in result["rows"]))
        self.assertNotIn("endpoint_uv", result["rows"][0])

    def test_legacy_error_dict_is_failed_not_successful_filter_output(self):
        spec = filter_spec(self.path)
        async def legacy_error(name, args):
            return ({"error": "iir_order is required"} if name == "filter_eeg" else {"ok": True}), None
        with patch.dict(tools.SESSION, {}, clear=True):
            result = asyncio.run(sweep.run_sweep(spec, legacy_error))
        self.assertTrue(all(not r["ok"] and "iir_order" in r["error"] for r in result["rows"]))
        self.assertTrue(all("diagnostics" not in r for r in result["rows"]))
        spec["base_pipeline"][1]["args"].pop("iir_order")
        self.assertIn("iir_order", " ".join(sweep.validate_sweep(spec)))

    def test_planner_accepts_final_step_without_endpoint_and_schema_allows_psd(self):
        spec = filter_spec(self.path)
        with patch.object(pipeline.ollama, "chat", return_value={"message": {"content": json.dumps(spec)}}):
            result = pipeline.propose_sweep("test", "Sweep low-pass 20 and 30 Hz; stop after filtering")
        self.assertTrue(result["_planner_meta"]["valid"])
        self.assertNotIn("endpoint", result)
        self.assertIn("compute_psd", pipeline._sweep_schema()["properties"]["endpoint"]["properties"]["tool"]["enum"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
