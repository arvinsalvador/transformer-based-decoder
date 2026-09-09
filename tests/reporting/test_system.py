"""System checks run inside the authoritative CPU Docker test environment."""

import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

import pytest

from src.config.settings import PROJECT_ROOT


@pytest.mark.parametrize(
    "script", sorted((PROJECT_ROOT / "scripts").glob("*.py")), ids=lambda path: path.name
)
def test_every_cli_help(script):
    result = subprocess.run(
        [sys.executable, str(script), "--help"], capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout


def test_real_streamlit_server_health(tmp_path):
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    env = {**os.environ, "CONFIG_PATH": "config/local.yaml", "DEVICE": "cpu"}
    with (tmp_path / "streamlit.log").open("w+") as log:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                "app/streamlit_app.py",
                "--server.address=127.0.0.1",
                f"--server.port={port}",
                "--browser.gatherUsageStats=false",
            ],
            cwd=PROJECT_ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        try:
            deadline = time.monotonic() + 45
            while time.monotonic() < deadline and process.poll() is None:
                try:
                    with urllib.request.urlopen(
                        f"http://127.0.0.1:{port}/_stcore/health", timeout=2
                    ) as response:
                        assert response.status == 200 and response.read() == b"ok"
                        return
                except (urllib.error.URLError, TimeoutError):
                    time.sleep(0.25)
            log.seek(0)
            pytest.fail(f"Streamlit failed to become healthy: {log.read()}")
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
