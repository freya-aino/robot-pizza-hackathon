"""
test_docker.py — build a temporary Dockerfile whose final command runs main.py.

Requires the Docker CLI to be installed and the daemon running.

Run:
    uv run pytest tests/test_docker.py -v
"""

import shutil
import subprocess
import tempfile
import textwrap
import uuid
from pathlib import Path

import pytest

MAIN_PY = Path(__file__).resolve().parents[1] / "src" / "robot_hackathon" / "main.py"

# A minimal Dockerfile that ends with running main.py.  Because main.py now
# imports LeRobot lazily, the image only needs a standard Python runtime.
DOCKERFILE = textwrap.dedent("""\
    FROM python:3.11-alpine
    WORKDIR /app
    COPY main.py .
    CMD ["python", "main.py", "--help"]
""")


@pytest.mark.integration
def test_temporary_dockerfile_runs_main_help():
    if shutil.which("docker") is None:
        pytest.skip("docker CLI is not available")

    tag = f"main-py-smoke:{uuid.uuid4().hex[:8]}"

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        shutil.copy(MAIN_PY, td / "main.py")
        (td / "Dockerfile").write_text(DOCKERFILE)

        try:
            subprocess.run(
                ["docker", "build", "-t", tag, "."],
                cwd=td,
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as e:
            pytest.fail(f"Docker build failed:\n{e.stdout}\n{e.stderr}")

        try:
            run = subprocess.run(
                ["docker", "run", "--rm", tag],
                capture_output=True,
                text=True,
                timeout=60,
                check=False,  # we assert the return code ourselves
            )
        finally:
            subprocess.run(
                ["docker", "rmi", "-f", tag],
                capture_output=True,
                text=True,
                check=False,  # cleanup failure must not mask the test result
            )

    assert run.returncode == 0, f"container exited {run.returncode}:\n{run.stderr}"
    assert "usage:" in run.stdout.lower()
