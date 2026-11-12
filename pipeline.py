"""Unified AESLC pipeline runner."""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.resolve()
PYTHON = ROOT.parent / ".venv" / ("Scripts" if sys.platform == "win32" else "bin") / "python"


def run_full():
    for script in ("preprocess.py", "train.py", "evaluate.py", "tests/smoke_test.py"):
        result = subprocess.run([str(PYTHON), str(ROOT / script)], cwd=ROOT)
        if result.returncode:
            raise SystemExit(result.returncode)
    print(f"Demo command: {PYTHON} {ROOT / 'app.py'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--app", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--personalize", action="store_true")
    args, remainder = parser.parse_known_args()
    if args.app:
        subprocess.run([str(PYTHON), str(ROOT / "app.py")], cwd=ROOT, check=True)
    elif args.test:
        subprocess.run([str(PYTHON), str(ROOT / "tests/smoke_test.py")], cwd=ROOT, check=True)
    elif args.personalize:
        from personalize import personalize
        print(personalize("Demo", "software engineering and energy markets", retrieval="tfidf"))
    else:
        run_full()


if __name__ == "__main__":
    main()
