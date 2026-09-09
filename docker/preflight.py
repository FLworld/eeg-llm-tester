#!/usr/bin/env python3
"""Check the actual Compose service's network, models, mounts and runtime."""
import json
import os
import subprocess
import sys
import tempfile
import urllib.request


def main():
    url = os.environ.get('OLLAMA_HOST', 'http://host.docker.internal:11434').rstrip('/')
    try:
        with urllib.request.urlopen(url + '/api/tags', timeout=10) as response:
            names = {m['name'] for m in json.load(response)['models']}
    except Exception as exc:
        print(f'FAIL: App container cannot reach Ollama at {url}: {exc}', flush=True)
        print('Open Ollama. If it is already running, follow Connection troubleshooting in START-HERE.html.\n'
              'Check the address, firewall and Docker networking; the bind address is only one possible cause.', flush=True)
        return 1
    print('PASS: App container can reach Ollama', flush=True)
    for name in ('eeg-qwen:latest', 'nomic-embed-text:latest'):
        if name not in names:
            print(f'FAIL: Model {name} missing. Run the Start launcher to prepare both models.', flush=True)
            return 1
        print(f'PASS: Model {name}', flush=True)
    for path in ('/data', '/output', '/state'):
        try:
            with tempfile.TemporaryFile(dir=path) as probe:
                probe.write(b'eeg-llm preflight')
                probe.flush()
        except OSError as exc:
            print(f'FAIL: Cannot write {path}: {exc}. Check Docker file sharing and folder permissions.', flush=True)
            return 1
    print('PASS: Recording, output and state folders are writable', flush=True)
    return subprocess.call([sys.executable, '/usr/local/bin/smoke.py'])


if __name__ == '__main__':
    sys.exit(main())
