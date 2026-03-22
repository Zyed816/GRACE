import os
import subprocess
import sys


if __name__ == "__main__":
    cmd = [
        sys.executable,
        os.path.join(os.path.dirname(__file__), "grid_search_gca_cora.py"),
        "--dataset",
        "DBLP",
        *sys.argv[1:],
    ]
    raise SystemExit(subprocess.call(cmd))
