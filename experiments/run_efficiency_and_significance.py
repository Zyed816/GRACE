import argparse
import os
import subprocess
import sys
from time import perf_counter as t


DATASET_CHOICES = ["Cora", "CiteSeer", "PubMed", "DBLP"]
METHOD_CHOICES = ["grace", "gca", "sg-gr", "sg-gc"]


def dedupe_keep_order(values):
    return list(dict.fromkeys(values))


def run_child(grace_dir, label, script_rel_path, child_args):
    cmd = [sys.executable, script_rel_path, *child_args]

    print("=" * 90)
    print(f"[extra-experiments] start: {label}")
    print(f"[extra-experiments] command: {' '.join(cmd)}")

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
        print(f"[{label}] {line.rstrip()}")

    proc.wait()
    elapsed = t() - start
    print(
        f"[extra-experiments] done: {label} | "
        f"code={proc.returncode} | elapsed={elapsed:.1f}s"
    )
    return proc.returncode


def build_common_args(datasets, methods, gpu_id, config):
    args = [
        "--datasets",
        *datasets,
        "--methods",
        *methods,
        "--gpu_id",
        str(gpu_id),
    ]
    if config:
        args.extend(["--config", config])
    return args


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run the full efficiency experiment followed by the paired-seed "
            "statistical significance experiment."
        )
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DATASET_CHOICES,
        choices=DATASET_CHOICES,
        help="Datasets used by both experiments.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=METHOD_CHOICES,
        choices=METHOD_CHOICES,
        help="Methods used by both experiments.",
    )
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--efficiency_runs", type=int, default=3)
    parser.add_argument("--significance_runs", type=int, default=10)
    parser.add_argument("--eval_repeats", type=int, default=3)
    parser.add_argument("--std_weight", type=float, default=0.5)
    parser.add_argument(
        "--continue_on_error",
        action="store_true",
        help="Forward continue_on_error to child scripts and continue to significance if efficiency fails.",
    )
    parser.add_argument(
        "--verbose_train_output",
        action="store_true",
        help="Forward verbose_train_output to both child scripts.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.efficiency_runs <= 0:
        raise RuntimeError("--efficiency_runs must be >= 1")
    if args.significance_runs <= 1:
        raise RuntimeError("--significance_runs must be >= 2")
    if args.eval_repeats <= 0:
        raise RuntimeError("--eval_repeats must be >= 1")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    grace_dir = os.path.abspath(os.path.join(script_dir, ".."))

    datasets = dedupe_keep_order(args.datasets)
    methods = dedupe_keep_order(args.methods)
    common = build_common_args(datasets, methods, args.gpu_id, args.config)
    shared_tail = ["--std_weight", str(args.std_weight)]
    if args.continue_on_error:
        shared_tail.append("--continue_on_error")
    if args.verbose_train_output:
        shared_tail.append("--verbose_train_output")

    efficiency_args = [
        *common,
        "--runs",
        str(args.efficiency_runs),
        *shared_tail,
    ]
    significance_args = [
        *common,
        "--runs",
        str(args.significance_runs),
        "--eval_repeats",
        str(args.eval_repeats),
        *shared_tail,
    ]

    total_start = t()
    failures = []

    efficiency_code = run_child(
        grace_dir=grace_dir,
        label="efficiency",
        script_rel_path=os.path.join("experiments", "efficiency", "run_efficiency_experiment.py"),
        child_args=efficiency_args,
    )
    if efficiency_code != 0:
        failures.append(("efficiency", efficiency_code))
        if not args.continue_on_error:
            print("[extra-experiments] stop: efficiency failed")
            raise SystemExit(1)

    significance_code = run_child(
        grace_dir=grace_dir,
        label="significance",
        script_rel_path=os.path.join(
            "experiments",
            "statistical_significance",
            "run_significance_experiment.py",
        ),
        child_args=significance_args,
    )
    if significance_code != 0:
        failures.append(("significance", significance_code))

    total_elapsed = t() - total_start
    print("=" * 90)
    print(f"[extra-experiments] all requested experiments finished in {total_elapsed:.1f}s")

    if failures:
        for label, code in failures:
            print(f"[extra-experiments] failed: {label}, code={code}")
        raise SystemExit(1)

    print("[extra-experiments] efficiency and significance experiments completed successfully")


if __name__ == "__main__":
    main()
