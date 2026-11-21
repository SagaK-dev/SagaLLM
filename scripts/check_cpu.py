"""Cross-platform, network-free post-installation CPU verification.

No paid API, cloud GPU, external dataset, or large-model allocation is used.
"""
from __future__ import annotations
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent


def run(*arguments: str) -> None:
    environment = os.environ.copy()
    environment.update({"OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "CUDA_VISIBLE_DEVICES": ""})
    environment["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + environment.get("PYTHONPATH", "")
    subprocess.run([sys.executable, *arguments], cwd=ROOT, env=environment, check=True)


def main() -> None:
    run("-m", "compileall", "-q", "src", "tests")
    run("-m", "pytest", "-q")
    run("-m", "sagallm", "doctor")
    for name in ("1b", "3b"):
        run("-m", "sagallm", "model", "inspect", "--config", f"configs/model/{name}.yaml")
    print("CPU CI checks passed; 1B/3B model weights were never allocated.")


if __name__ == "__main__":
    main()