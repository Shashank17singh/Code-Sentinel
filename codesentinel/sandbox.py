"""Run untrusted candidate code in a restricted subprocess.

Protections: separate interpreter in isolated mode (``-I``), empty temp working
directory, scrubbed environment (API keys are NOT inherited), wall-clock timeout,
output truncation and, on POSIX, CPU / memory / file-size limits that the code
cannot raise again (soft limit == hard limit).

This is defence in depth for a triage tool, not a hardened sandbox. When running
untrusted code from the public internet, put the whole app in a container or VM.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

from codesentinel.config import RUN_MAX_OUTPUT_CHARS, RUN_MEMORY_MB, RUN_TIMEOUT_SECONDS

# Executed inside the child interpreter: apply limits first, then run the candidate file.
# (Limits are set in the child rather than via preexec_fn, which is unsafe with threads.)
_LAUNCHER = """
import runpy, sys
try:
    import resource
    mem = {mem} * 1024 * 1024
    for name, value in (("RLIMIT_AS", mem), ("RLIMIT_CPU", {cpu}), ("RLIMIT_FSIZE", 1024 * 1024)):
        resource.setrlimit(getattr(resource, name), (value, value))
except ImportError:  # Windows: no resource module
    pass
runpy.run_path(sys.argv[1], run_name="__main__")
"""


def _clean_env() -> dict[str, str]:
    env = {"PYTHONIOENCODING": "utf-8", "PATH": os.environ.get("PATH", "")}
    for key in ("SYSTEMROOT", "TEMP", "TMP"):  # required by Python on Windows
        if key in os.environ:
            env[key] = os.environ[key]
    return env


def run_code(code: str, stdin: str = "", timeout: int = RUN_TIMEOUT_SECONDS) -> str:
    """Execute ``code`` and return stdout, or the sentinels ``ERROR`` / ``TIMEOUT``."""
    launcher = _LAUNCHER.format(mem=RUN_MEMORY_MB, cpu=timeout + 1)
    with tempfile.TemporaryDirectory() as workdir:
        script = Path(workdir) / "candidate.py"
        script.write_text(code, encoding="utf-8")
        try:
            result = subprocess.run(
                [sys.executable, "-I", "-c", launcher, str(script)],
                input=stdin,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=workdir,
                env=_clean_env(),
            )
        except subprocess.TimeoutExpired:
            return "TIMEOUT"
        except OSError:
            return "ERROR"
    if result.returncode != 0:
        return "ERROR"
    return result.stdout.strip()[:RUN_MAX_OUTPUT_CHARS]
