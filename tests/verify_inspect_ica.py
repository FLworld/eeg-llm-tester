"""Deterministic ICA inspection routing and real synthetic-data plot regressions."""
import asyncio
import io
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import app
import tools
import mne
import numpy as np
from PIL import Image


def synthetic_ica():
    rng = np.random.default_rng(42)
    names = ["Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4"]
    sources = rng.laplace(size=(8, 2560))
    raw = mne.io.RawArray(rng.normal(size=(8, 8)) @ sources * 1e-6,
                         mne.create_info(names, 128, "eeg"), verbose="ERROR")
    raw.set_montage("standard_1020")
    raw.filter(1, 40, verbose="ERROR")
    ica = mne.preprocessing.ICA(n_components=8, method="infomax", random_state=42,
                                max_iter=100, verbose="ERROR")
    ica.fit(raw, verbose="ERROR")
    return raw, ica


class Message:
    sent = []

    def __init__(self, content, elements=None, **kwargs):
        self.content = content
        self.elements = elements or []

    async def send(self):
        self.sent.append(self)

    async def update(self):
        pass


class Step:
    def __init__(self, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    async def update(self):
        pass


class InspectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw, cls.ica = synthetic_ica()

    def setUp(self):
        self.session = {"history": [], "applied_steps": []}
        Message.sent = []
        def make_async(fn):
            async def invoke(*args, **kwargs):
                return fn(*args, **kwargs)
            return invoke
        for target, value in [
            ("Message", Message), ("Step", Step), ("Image", SimpleNamespace), ("make_async", make_async),
            ("user_session", SimpleNamespace(get=self.session.get, set=self.session.__setitem__)),
        ]:
            p = patch.object(app.cl, target, value)
            p.start()
            self.addCleanup(p.stop)
        p = patch.object(app, "_handle_agentic", side_effect=AssertionError("Must not call the model"))
        p.start()
        self.addCleanup(p.stop)

    def send(self, prompt):
        asyncio.run(app.on_message(Message(prompt)))
        return Message.sent[0]

    def test_exact_request_renders_both_components_without_modifying_data(self):
        before = self.raw.get_data().copy()
        exclude = self.ica.exclude.copy()
        with patch.dict(tools.SESSION, {"raw": self.raw, "ica": self.ica, "epochs": None}, clear=True):
            out = self.send("inspect components 1 and 8")
        self.assertIn("ICA components 1, 8 (1-based)", out.content)
        self.assertEqual(len(out.elements), 1)
        im = Image.open(io.BytesIO(out.elements[0].content)).convert("RGB")
        self.assertGreater(im.height, im.width)
        self.assertGreater(np.asarray(im).std(), 10)
        np.testing.assert_array_equal(before, self.raw.get_data())
        self.assertEqual(exclude, self.ica.exclude)
        self.assertEqual(self.session["history"][-1]["content"], out.content)

    def test_no_ica_and_out_of_range_are_visible(self):
        for ica, prompt, error in [(None, "inspect components 1 and 8", "No ICA in session"),
                                   (self.ica, "/inspect-ica 1 9", "out of range 1..8"),
                                   (self.ica, "/inspect-ica 0", "out of range 1..8")]:
            Message.sent = []
            with patch.dict(tools.SESSION, {"ica": ica}, clear=True):
                out = self.send(prompt)
            self.assertIn(error, out.content)
            self.assertEqual(out.elements, [])

    def test_aliases_preserve_one_based_numbers(self):
        for prompt in ["/inspect-ica 1 8", "/inspect-ica [1, 8]", "inspect ICA components 1 and 8",
                       "Please inspect components 1, 8.", "show components 1 & 8"]:
            Message.sent = []
            with patch.object(app, "call_tool", return_value={"ok": False, "error": "test"}) as call:
                self.send(prompt)
            call.assert_called_once_with("inspect_ica_component", {"components": [1, 8]})

    def test_invalid_lists_do_not_run_tools(self):
        for prompt in ["/inspect-ica", "inspect components", "/inspect-ica 1.8", "/inspect-ica -1",
                       "/inspect-ica 1 to 8", "/inspect-ica 1 2 3 4 5 6 7"]:
            Message.sent = []
            with patch.object(app, "call_tool") as call:
                out = self.send(prompt)
            call.assert_not_called()
            self.assertTrue(out.content)

    def test_compound_actions_and_parameter_questions_stay_on_existing_route(self):
        for prompt in ["/plan inspect components 1 and 8", "inspect components 1 and 8 then remove 1",
                       "do not inspect components 1 and 8", "what parameters does inspect_ica_component take?",
                       "inspect components 1 or 8", "inspect components 1 through 8"]:
            self.assertIsNone(app._inspect_components_request(prompt), prompt)

    def test_missing_image_is_visible(self):
        with patch.object(app, "call_tool", return_value={"ok": True}):
            self.assertIn("returned no plot", self.send("/inspect-ica 1").content)


if __name__ == "__main__":
    unittest.main(verbosity=2)
