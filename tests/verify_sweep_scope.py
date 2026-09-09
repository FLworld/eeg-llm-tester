"""Existing/input-root codebooks, compound scope requests, and reused epoch windows."""
import asyncio
import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import app
import mne
import numpy as np
import pipeline
import sweep
import tools
from verify_inspect_ica import Message, Step
from verify_sweep_prerequisites import GOOD, REQUEST

CODEBOOK = {"conditions": {"face": [[1, 40]], "car": [[41, 80]]}, "responses": {"correct": 201}}


class ScopeSweepTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = self.root / "data"
        self.inputs = self.root / "data-in"
        self.inputs.mkdir()
        self.recording = self.data / "sub-002/eeg/sub-002_task-N170_eeg.fif"
        self.recording.parent.mkdir(parents=True)
        raw = mne.io.RawArray(np.zeros((2, 1280)), mne.create_info(["PO8", "PO7"], 128, "eeg"), verbose="ERROR")
        raw.set_annotations(mne.Annotations([1, 3, 5, 7], [0] * 4, ["1", "41", "1", "41"]))
        raw.save(self.recording, verbose="ERROR")
        self.cb_name = "sub-002_task-N170_eeg.codebook.json"
        self.session = {}
        Message.sent = []
        def make_async(fn):
            async def invoke(*args, **kwargs):
                return fn(*args, **kwargs)
            return invoke
        for target, key, value in [
            (app.cl, "Message", Message), (app.cl, "Step", Step), (app.cl, "make_async", make_async),
            (app.cl, "user_session", SimpleNamespace(get=self.session.get, set=self.session.__setitem__)),
            (app, "DATA_DIR", str(self.data)), (tools, "DATA_DIR", str(self.data)),
            (app, "retrieve_context", lambda request: {"text": ""}),
        ]:
            p = patch.object(target, key, value)
            p.start()
            self.addCleanup(p.stop)
        for mapping in (tools.SESSION, tools.SCOPE):
            p = patch.dict(mapping, {}, clear=True)
            p.start()
            self.addCleanup(p.stop)

    def test_existing_data_in_codebook_is_found_without_upload(self):
        (self.inputs / self.cb_name).write_text(json.dumps(CODEBOOK))
        config = copy.deepcopy(GOOD)
        config["base_pipeline"][-1]["args"].update(tmin=.11, tmax=.15)
        with patch.object(pipeline.ollama, "chat", return_value={"message": {"content": json.dumps(config)}}) as chat:
            asyncio.run(app._handle_sweep(f"scope {self.recording.name}, " + REQUEST))
        self.assertEqual(tools.SCOPE["codebook"], CODEBOOK)
        self.assertEqual(tools.SESSION["filepath"], str(self.recording))
        staged = self.session["pending_sweep"]
        self.assertEqual(staged["base_pipeline"][-1]["args"]["tmin"], -.2)
        self.assertEqual(staged["endpoint"]["args"]["tmin"], .11)
        self.assertEqual(chat.call_count, 1)
        self.assertIn("component measurement window is unchanged", " ".join(staged["notes"]))

    def test_uploaded_codebook_is_applied_after_same_message_scope(self):
        uploaded = self.root / self.cb_name
        uploaded.write_text(json.dumps(CODEBOOK))
        msg = SimpleNamespace(content=f"/sweep scope {self.recording.name}, " + REQUEST,
                              elements=[SimpleNamespace(name=self.cb_name, path=str(uploaded))])
        with patch.object(pipeline.ollama, "chat", return_value={"message": {"content": json.dumps(GOOD)}}):
            asyncio.run(app.on_message(msg))
        self.assertEqual(tools.SCOPE["codebook"], CODEBOOK)
        self.assertIsNone(self.session.get("pending_codebook_uploads"))
        self.assertTrue(Path(tools._codebook_path(str(self.recording))).exists())
        self.assertIn("Applied codebook", " ".join(m.content for m in Message.sent))
        self.assertIsNotNone(self.session.get("pending_sweep"))

    def test_adjacent_codebook_has_priority_and_unrelated_names_are_not_guessed(self):
        (self.inputs / self.cb_name).write_text(json.dumps(CODEBOOK))
        own = {"conditions": {"different_task": [1]}}
        sidecar = Path(tools._codebook_path(str(self.recording)))
        sidecar.write_text(json.dumps(own))
        self.assertEqual(tools._load_codebook_file(str(self.recording)), own)
        sidecar.unlink()
        (self.inputs / self.cb_name).rename(self.inputs / "sub-999.codebook.json")
        self.assertIsNone(tools._load_codebook_file(str(self.recording)))

    def test_bad_scope_cannot_reuse_a_staged_plan_or_call_model(self):
        self.session.update(pending_sweep=GOOD, pending_pipeline={"old": True})
        with patch.object(pipeline.ollama, "chat") as chat:
            asyncio.run(app._handle_sweep("scope nonexistent.fif, " + REQUEST))
        chat.assert_not_called()
        self.assertIsNone(self.session["pending_sweep"])
        self.assertIsNone(self.session["pending_pipeline"])
        self.assertIn("Scope failed", Message.sent[-1].content)

    def test_variant_download_declares_json_mime_type(self):
        row = {"ok": True, "label": "filter_eeg.h_freq=30", "assignments": {"filter_eeg.h_freq": 30},
               "output_tool": "filter_eeg", "output": {"ok": True, "h_freq": 30}, "metrics": {}}
        async def completed(*args):
            return {"ok": True, "rows": [row], "n_variants": 1}
        with patch.object(app, "run_sweep", completed), patch.object(app, "_plot_spec_curve", return_value=None), \
                patch.object(app.cl, "File", SimpleNamespace):
            asyncio.run(app._run_sweep_and_render({}))
        attachment = Message.sent[-1].elements[0]
        self.assertEqual(attachment.mime, "application/json")
        self.assertEqual(json.loads(attachment.content), row)

    def test_scope_prefix_is_explicit_and_preserves_quoted_paths(self):
        self.assertEqual(app._sweep_scope_prefix('scope "a recording.set", filter it'), ("a recording.set", "filter it"))
        self.assertEqual(app._sweep_scope_prefix("scope 'a,b.set'; filter it"), ("a,b.set", "filter it"))
        self.assertEqual(app._sweep_scope_prefix("load it and scope later"), (None, "load it and scope later"))

    def test_plural_aliases_keep_codes_and_exact_labels_take_priority(self):
        self.assertEqual(sweep.validate_sweep_request(GOOD, REQUEST, {"codebook": CODEBOOK}), [])
        self.assertEqual(tools.resolve_condition_codes("FACES", CODEBOOK["conditions"]), [[1, 40]])
        self.assertEqual(tools.resolve_condition_codes("faces", {"face": [1], "faces": [2]}), [2])
        self.assertIsNone(tools.resolve_condition_codes("unrelated", CODEBOOK["conditions"]))
        bad = copy.deepcopy(GOOD)
        bad["base_pipeline"][3]["args"]["bins"][0]["codes"] = [[101, 140]]
        self.assertTrue(sweep.validate_sweep_request(bad, REQUEST, {"codebook": CODEBOOK}))

    def test_window_normalization_does_not_guess_for_raw_events_or_swept_windows(self):
        for change in ("raw_labels", "window_axis"):
            spec = copy.deepcopy(GOOD)
            args = spec["base_pipeline"][-1]["args"]
            args.update(tmin=.11, tmax=.15)
            if change == "raw_labels":
                args.update(event_id_a="1", event_id_b="41")
            else:
                spec["axes"].append({"tool": "create_epochs", "param": "tmin", "values": [-.2, -.1]})
            pipeline._sanitize_sweep(spec)
            self.assertEqual(args["tmin"], .11)
            self.assertTrue(sweep.validate_sweep(spec))


if __name__ == "__main__":
    unittest.main(verbosity=2)
