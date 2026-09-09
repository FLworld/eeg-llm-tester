"""Failure-path tests with fake host commands; no downloads or running app affected."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from io import BytesIO

ROOT = Path(__file__).resolve().parents[2]
FAKE = '''#!/usr/bin/python3
import json, os, sys
from pathlib import Path
name = Path(sys.argv[0]).name
args = sys.argv[1:]
step = name + ' ' + ' '.join(args)
with open(os.environ['TRACE'], 'a') as f:
    f.write(json.dumps({'step':step,'host':os.environ.get('OLLAMA_HOST')})+'\\n')
if step == os.environ.get('FAIL_STEP'):
    print('simulated failure', file=sys.stderr)
    sys.exit(1)
if step == 'docker info --format {{.OSType}}': print(os.environ.get('SERVER_OS','linux'))
if step == 'docker compose config --images': print(os.environ.get('FAKE_IMAGE','eeg-llm:latest'))
if step == 'docker compose port eeg-llm 8001': print('127.0.0.1:18091')
'''


class LauncherTests(unittest.TestCase):
    def run_launcher(self, fail='', flag='', **extra):
        with tempfile.TemporaryDirectory(prefix='eeg setup with spaces ') as tmp:
            folder = Path(tmp)
            shutil.copy2(ROOT / 'start-eeg-llm.sh', folder)
            shutil.copy2(ROOT / 'Start eeg-llm.command', folder)
            bindir = folder / 'bin'
            bindir.mkdir()
            for name in ('docker', 'ollama'):
                target = bindir / name
                target.write_text(FAKE)
                target.chmod(0o755)
            trace = folder / 'trace'
            env = dict(os.environ, PATH=f'{bindir}:/usr/bin:/bin', TRACE=str(trace),
                       EEG_NO_BROWSER='1', FAIL_STEP=fail, **extra)
            cmd = ['bash', str(folder / 'Start eeg-llm.command')]
            if flag:
                cmd.append(flag)
            result = subprocess.run(cmd, cwd='/', env=env, text=True, capture_output=True)
            logs = [json.loads(line) for line in trace.read_text().splitlines()]
            return result, logs

    def test_success_and_paths_with_spaces(self):
        result, logs = self.run_launcher(OLLAMA_HOST='http://wrong-host:11434')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('Ready: http://127.0.0.1:18091', result.stdout)
        self.assertIn('docker compose build eeg-llm', [x['step'] for x in logs])
        for item in logs:
            if item['step'].startswith('ollama '):
                self.assertEqual(item['host'], 'http://127.0.0.1:11434')
            if item['step'].startswith('docker compose run '):
                self.assertIsNone(item['host'])

    def test_hard_failures_never_report_ready(self):
        for fail in ('docker compose version', 'docker info', 'ollama list',
                     'ollama create eeg-qwen -f Modelfile', 'docker compose build eeg-llm',
                     'docker compose run --rm --no-deps --entrypoint python eeg-llm /usr/local/bin/preflight.py',
                     'docker compose up -d --no-build --wait --wait-timeout 180'):
            with self.subTest(fail=fail):
                result, _ = self.run_launcher(fail=fail)
                self.assertNotEqual(result.returncode, 0)
                self.assertNotIn('Ready:', result.stdout)

    def test_missing_embedding_is_pulled(self):
        result, logs = self.run_launcher(fail='ollama show nomic-embed-text:latest')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('ollama pull nomic-embed-text:latest', [x['step'] for x in logs])

    def test_check_does_not_build_download_or_start(self):
        result, logs = self.run_launcher(flag='--check')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('CHECK: PASS', result.stdout)
        self.assertFalse(any((' pull ' in x['step'] or ' build ' in x['step'] or ' up ' in x['step']) for x in logs))

    def test_windows_containers_rejected(self):
        result, _ = self.run_launcher(SERVER_OS='windows')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Linux containers', result.stderr)

    def test_prebuilt_image_uses_effective_compose_configuration(self):
        result, logs = self.run_launcher(FAKE_IMAGE='local-offline:test')
        self.assertEqual(result.returncode, 0, result.stderr)
        steps = [x['step'] for x in logs]
        self.assertIn('docker image inspect local-offline:test', steps)
        self.assertNotIn('docker compose build eeg-llm', steps)
        self.assertNotIn('docker compose pull eeg-llm', steps)

    def test_stop_does_not_require_ollama(self):
        result, logs = self.run_launcher(flag='--stop', fail='ollama list')
        self.assertEqual(result.returncode, 0)
        self.assertFalse(any(x['step'].startswith('ollama') for x in logs))


spec = importlib.util.spec_from_file_location('preflight', ROOT / 'docker/preflight.py')
preflight = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preflight)


class PreflightTests(unittest.TestCase):
    def check(self, names, smoke=0, network_error=None, mount_error=None):
        response = BytesIO(json.dumps({'models': [{'name': x} for x in names]}).encode())
        with patch.object(preflight.urllib.request, 'urlopen', return_value=response, side_effect=network_error), \
             patch.object(preflight.tempfile, 'TemporaryFile', side_effect=mount_error), \
             patch.object(preflight.subprocess, 'call', return_value=smoke) as call:
            result = preflight.main()
            return result, call.called

    def test_both_models_and_runtime_required(self):
        self.assertEqual(self.check(['eeg-qwen:latest', 'nomic-embed-text:latest']), (0, True))
        self.assertEqual(self.check(['eeg-qwen:latest', 'nomic-embed-text:latest'], smoke=1), (1, True))

    def test_wrong_tag_and_prefix_do_not_count(self):
        for names in (['eeg-qwen:old', 'nomic-embed-text:latest'], ['eeg-qwen-extra:latest', 'nomic-embed-text:latest'], ['eeg-qwen:latest']):
            self.assertEqual(self.check(names), (1, False))

    def test_network_and_storage_fail_closed(self):
        self.assertEqual(self.check([], network_error=TimeoutError('test')), (1, False))
        self.assertEqual(self.check(['eeg-qwen:latest', 'nomic-embed-text:latest'], mount_error=PermissionError('test')), (1, False))


if __name__ == '__main__':
    unittest.main()
