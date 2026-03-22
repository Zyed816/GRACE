import argparse
import os
import subprocess
import sys
from time import perf_counter as t


DATASET_SCRIPT = {
    "Cora": "run_cora_full_pipeline.py",
    "CiteSeer": "run_citeseer_full_pipeline.py",
    "PubMed": "run_pubmed_full_pipeline.py",
    "DBLP": "run_dblp_full_pipeline.py",
}


def run_one(grace_dir, dataset, script_name, child_args):
    script_path = os.path.join(grace_dir, "tools", script_name)
    cmd = [sys.executable, script_path, *child_args]

    print("=" * 90)
    print(f"[dispatch] dataset={dataset} | script={script_name}")
    print(f"[dispatch] command: {' '.join(cmd)}")

    start = t()
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.Popen(
        cmd,
        cwd=grace_dir,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
        env=env,
    )

    assert proc.stdout is not None
    for line in proc.stdout:
        print(f"[{dataset}] {line.rstrip()}")

    proc.wait()
    elapsed = t() - start

    print(f"[dispatch] dataset={dataset} finished with code={proc.returncode} in {elapsed:.1f}s")
    return proc.returncode


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run selected dataset full-pipeline scripts sequentially. "
            "Unknown args are forwarded to each child script."
        )
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["Cora", "CiteSeer", "PubMed", "DBLP"],
        choices=["Cora", "CiteSeer", "PubMed", "DBLP"],
        help="Datasets to run in order.",
    )
    parser.add_argument(
        "--continue_on_error",
        action="store_true",
        help="Continue to next dataset when a child pipeline fails.",
    )

    args, passthrough = parser.parse_known_args()

    tools_dir = os.path.dirname(os.path.abspath(__file__))
    grace_dir = os.path.abspath(os.path.join(tools_dir, ".."))

    # Keep order while removing duplicates in --datasets.
    datasets = list(dict.fromkeys(args.datasets))

    failures = []
    total_start = t()

    for dataset in datasets:
        code = run_one(grace_dir, dataset, DATASET_SCRIPT[dataset], passthrough)
        if code != 0:
            failures.append((dataset, code))
            if not args.continue_on_error:
                break

    total_elapsed = t() - total_start
    print("=" * 90)
    print(f"[dispatch] all requested runs finished in {total_elapsed:.1f}s")

    if failures:
        for dataset, code in failures:
            print(f"[dispatch] failed: dataset={dataset}, code={code}")
        raise SystemExit(1)

    print("[dispatch] all datasets completed successfully")


if __name__ == "__main__":
    main()
