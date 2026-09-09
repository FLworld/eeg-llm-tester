"""Headed real-plot verification in an isolated synthetic-ICA server, or live no-ICA smoke."""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from playwright.sync_api import sync_playwright


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--empty-session-url", help="Only inspect; expect no ICA. Never loads or fits data.")
    opts = parser.parse_args()
    opts.output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[2]
    process = None
    log = None
    results = []
    try:
        url = opts.empty_session_url
        if not url:
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0))
                port = sock.getsockname()[1]
            log = (opts.output / "fixture-server.log").open("w")
            process = subprocess.Popen([
                sys.executable, "-m", "chainlit", "run", "tests/ui/inspect_ica_fixture.py",
                "--host", "127.0.0.1", "--port", str(port), "--headless",
            ], cwd=root, stdout=log, stderr=subprocess.STDOUT,
                env=dict(os.environ, STATE_DIR=str(opts.output / "state")))
            deadline = time.monotonic() + 60
            while True:
                try:
                    with socket.create_connection(("127.0.0.1", port), timeout=1):
                        break
                except OSError:
                    if process.poll() is not None or time.monotonic() > deadline:
                        raise RuntimeError("Fixture server did not start; inspect fixture-server.log")
                    time.sleep(0.3)
            url = f"http://127.0.0.1:{port}"
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=False)
            context = browser.new_context(viewport={"width": 1440, "height": 1100})
            context.tracing.start(screenshots=True, snapshots=True, sources=True)
            page = context.new_page()
            page.route("**/*.{woff,woff2,ttf,otf,eot}", lambda route: route.abort())
            try:
                page.goto(url, wait_until="domcontentloaded")
                page.locator('[data-step-type="assistant_message"]').first.wait_for(timeout=45000)
                page.wait_for_timeout(800)
                cases = [("inspect components 1 and 8", "No ICA in session", False)] if opts.empty_session_url else [
                    ("inspect components 1 and 8", "ICA components 1, 8 (1-based)", True),
                    ("/inspect-ica 1 8", "ICA components 1, 8 (1-based)", True),
                    ("/inspect-ica 9", "out of range 1..8", False),
                ]
                for i, (prompt, expected, has_image) in enumerate(cases, 1):
                    replies = page.locator('[data-step-type="assistant_message"]').filter(has_text=expected)
                    count = replies.count()
                    page.locator("textarea").last.fill(prompt)
                    page.locator("textarea").last.press("Enter")
                    reply = replies.nth(count)
                    reply.wait_for(timeout=60000)
                    row = {"prompt": prompt, "response": reply.inner_text(), "synthetic": not bool(opts.empty_session_url)}
                    if has_image:
                        img = reply.locator('img:not([alt^="Avatar"])').first
                        img.wait_for(timeout=15000)
                        img.evaluate("img => img.decode()")
                        row["image"] = img.evaluate("img => ({width: img.naturalWidth, height: img.naturalHeight})")
                        assert row["image"]["height"] > row["image"]["width"] > 100
                        img.screenshot(path=str(opts.output / f"{i}-plot.png"))
                    page.screenshot(path=str(opts.output / f"{i}-page.png"), full_page=True)
                    row["status"] = "passed"
                    results.append(row)
                    print("PASS", prompt, flush=True)
                    page.wait_for_timeout(600)
            finally:
                (opts.output / "results.json").write_text(json.dumps(results, indent=2))
                (opts.output / "transcript.txt").write_text(page.locator("body").inner_text())
                context.tracing.stop(path=str(opts.output / "trace.zip"))
                context.close()
                browser.close()
    finally:
        if process:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        if log:
            log.close()


if __name__ == "__main__":
    main()
