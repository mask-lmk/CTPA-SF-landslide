"""Run model/seed pairs sequentially on one GPU, with a log for each run."""

import argparse
import csv
import os
import subprocess
import sys
from datetime import datetime
from itertools import product
from pathlib import Path
from time import perf_counter

from yaml import safe_load

from run_study import SF_VARIANTS


MODELS = ("yolo11n", "ctpa", "sf", *SF_VARIANTS)


def command(args, model, seed):
    return [
        sys.executable, "-u", str(Path(__file__).with_name("run_study.py")),
        "--repo", str(args.repo), "--data", str(args.data), "--output", str(args.output),
        "--models", model, "--seeds", str(seed), "--epochs", str(args.epochs),
        "--patience", str(args.patience), "--batch", str(args.batch),
    ]


def invoke(argv, log_path):
    with log_path.open("w", encoding="utf-8") as log:
        return subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT, check=False).returncode


def training_status(args, folder):
    if not folder.exists():
        return "ready"
    if not ((folder / "completed.json").is_file() and
            (folder / "weights" / "best.pt").is_file() and
            (folder / "args.yaml").is_file()):
        return "partial_run"
    settings = safe_load((folder / "args.yaml").read_text(encoding="utf-8"))
    same_data = Path(settings["data"]).resolve() == args.data
    same_schedule = all(settings[key] == getattr(args, key)
                        for key in ("epochs", "patience", "batch"))
    return "already_trained" if same_data and same_schedule else "different_training_settings"


def run_queue(args):
    models = MODELS if args.models == ["all"] else list(dict.fromkeys(args.models))
    if "all" in models:
        raise ValueError("Use either --models all or a list of individual model names.")
    jobs = list(product(dict.fromkeys(args.seeds), models))
    print("{} jobs, one GPU job at a time".format(len(jobs)), flush=True)
    if args.dry_run:
        for seed, model in jobs:
            folder = args.output / "{}_seed{}".format(model, seed)
            print("{} seed={} [{}]".format(model, seed, training_status(args, folder)))
        return 0

    logs = args.output / "queues" / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    logs.mkdir(parents=True)
    print("Logs and status: " + str(logs), flush=True)
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    failures = 0
    with (logs / "status.csv").open("w", newline="", encoding="utf-8") as status:
        report = csv.writer(status)
        report.writerow(("model", "seed", "status", "duration_seconds", "train_log", "evaluation_log"))
        status.flush()
        for index, (seed, model) in enumerate(jobs, 1):
            name = "{}_seed{}".format(model, seed)
            print("[{}/{}] {}".format(index, len(jobs), name), flush=True)
            start = perf_counter()
            train_log = logs / (name + "_train.log")
            eval_log = logs / (name + "_evaluate.log")
            argv = command(args, model, seed)
            outcome = training_status(args, args.output / name)
            if outcome == "ready":
                outcome = "trained" if invoke(argv, train_log) == 0 else "train_failed"

            if outcome in ("trained", "already_trained") and not args.train_only:
                result = (args.output / "evaluation" /
                          "{}_{}_{}".format(args.data.stem, args.split, name) / "metrics.json")
                if result.is_file():
                    outcome = "already_evaluated"
                else:
                    code = invoke(argv + ["--stage", "evaluate", "--split", args.split], eval_log)
                    outcome = "evaluated" if code == 0 else "evaluation_failed"

            failures += outcome in ("partial_run", "train_failed", "evaluation_failed",
                                    "different_training_settings")
            report.writerow((model, seed, outcome, round(perf_counter() - start, 1),
                             train_log.name if train_log.exists() else "",
                             eval_log.name if eval_log.exists() else ""))
            status.flush()
            print("  " + outcome, flush=True)

    print("Completed queue with {} incomplete jobs.".format(failures), flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, type=Path, help="YOLO fork with CTPA modules")
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--models", nargs="+", required=True, choices=("all", *MODELS))
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.repo = args.repo.resolve(strict=True)
    args.data = args.data.resolve(strict=True)
    args.output = args.output.resolve()
    sys.exit(run_queue(args))
