"""Executa o fluxo local com o corpus existente.

Uso:
    python run_pipeline.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def run(cmd: list[str]) -> None:
    print("\n$ " + " ".join(cmd))
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> int:
    run([sys.executable, "src/generate_runs.py", "--top-k", "10"])
    run([sys.executable, "src/evaluate.py", "--k", "10"])
    run([sys.executable, "src/demo.py", "generative AI in education", "--top-k", "5"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
