import os
import subprocess
import sys


if __name__ == "__main__":
    cmd = [
        sys.executable,
        os.path.join(os.path.dirname(__file__), "run_cora_full_pipeline.py"),
        "--dataset",
        "CiteSeer",
        *sys.argv[1:],
    ]
    raise SystemExit(subprocess.call(cmd))
