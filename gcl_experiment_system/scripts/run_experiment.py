import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml


def dataset_key(dataset: str) -> str:
    return "dblp" if dataset.upper() == "DBLP" else dataset


def main():
    parser = argparse.ArgumentParser(description="Run a GRACE experiment with temporary config overrides")
    parser.add_argument("--dataset", required=True, choices=["Cora", "CiteSeer", "PubMed", "DBLP"])
    parser.add_argument("--model", required=True, choices=["grace", "gca", "ifl-gr", "ifl-gc"])
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--drop_edge_rate", type=float, default=0.2)
    parser.add_argument("--drop_feature_rate", type=float, default=0.2)
    parser.add_argument("--config", type=str, default=str(Path(__file__).resolve().parents[2] / "config.yaml"))
    parser.add_argument("--train_script", type=str, default=str(Path(__file__).resolve().parents[2] / "train.py"))
    parser.add_argument("--exp1_log_csv", type=str, default="")
    args = parser.parse_args()

    repo_root = Path(args.train_script).resolve().parent
    with open(args.config, "r", encoding="utf-8") as fp:
        cfg = yaml.safe_load(fp)

    section = cfg[dataset_key(args.dataset)]
    section["learning_rate"] = args.learning_rate
    section["num_hidden"] = args.hidden_dim
    section["num_proj_hidden"] = args.hidden_dim
    section["tau"] = args.temperature
    section["num_epochs"] = args.epochs
    section["drop_edge_rate_1"] = args.drop_edge_rate
    section["drop_edge_rate_2"] = args.drop_edge_rate
    section["drop_feature_rate_1"] = args.drop_feature_rate
    section["drop_feature_rate_2"] = args.drop_feature_rate

    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8") as fp:
        yaml.safe_dump(cfg, fp, sort_keys=False)
        temp_config = fp.name

    cmd = [
        sys.executable,
        args.train_script,
        "--dataset",
        args.dataset,
        "--method",
        args.model,
        "--config",
        temp_config,
    ]
    if args.exp1_log_csv:
        cmd.extend(["--exp1_log_csv", args.exp1_log_csv])

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    proc = subprocess.run(cmd, cwd=repo_root, text=True, capture_output=True, env=env)
    payload = {
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    raise SystemExit(proc.returncode)


if __name__ == "__main__":
    main()
