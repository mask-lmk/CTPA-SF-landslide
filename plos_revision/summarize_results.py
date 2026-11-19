"""Export per-seed metrics, descriptive statistics, and paired differences."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


def summarize(studies, output):
    metrics = ["precision", "recall", "map50", "map5095"]
    columns = ["study", "data", "model", "seed", "split", *metrics, "parameters"]
    rows = []
    grouped = defaultdict(dict)
    for path in studies:
        for report in sorted((path / "evaluation").glob("*/metrics.json")):
            row = dict(study=path.name, **json.loads(report.read_text(encoding="utf-8")))
            key = (row["study"], row["data"], row["split"], row["model"])
            if row["seed"] in grouped[key]:
                raise ValueError("Duplicate evaluation for {} seed {}".format(key, row["seed"]))
            grouped[key][row["seed"]] = row
            rows.append(row)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "per_seed_metrics.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    summary = []
    for (study, data, split, model), runs in sorted(grouped.items()):
        if set(runs) != {0, 1, 2}:
            continue
        result = dict(study=study, data=data, split=split, model=model, seeds=3)
        for key in metrics:
            values = [runs[seed][key] * 100 for seed in (0, 1, 2)]
            result[key] = {"mean": mean(values), "sd": stdev(values)}
        reference = "sf" if model.startswith("sf_") else "yolo11n"
        control = grouped.get((study, data, split, reference), {})
        if model != reference and set(control) == set(runs):
            differences = [(runs[seed]["map5095"] - control[seed]["map5095"]) * 100 for seed in (0, 1, 2)]
            result["paired_map5095"] = dict(control=reference, differences=differences,
                                          mean=mean(differences), sd=stdev(differences))
        summary.append(result)
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summarize(args.study, args.output)
