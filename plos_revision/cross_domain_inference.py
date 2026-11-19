"""Fixed exploratory inference comparisons on previously inspected external sets."""

import argparse
import csv
import json
import os
import sys
from pathlib import Path


def run(args):
    os.environ.setdefault("YOLO_CONFIG_DIR", str(args.output / "settings"))
    os.environ.setdefault("YOLO_AUTOINSTALL", "false")
    sys.path.insert(0, str(args.repo))
    from ultralytics import YOLO

    configurations = [("tta640", 640, True), ("single832", 832, False)]
    args.output.mkdir(parents=True, exist_ok=True)
    protocol = {
        "purpose": "Exploratory follow-up on previously inspected external test sets",
        "training_source": "MLD", "models": ["yolo11n", "sf"], "seeds": [0, 1, 2],
        "datasets": ["bijie", "cas_moxi"],
        "configurations": configurations, "conf": .001, "nms_iou": .7, "max_det": 300,
        "tta_scales": [1, .83, .67], "tta_flips": [None, "horizontal", None],
        "checkpoint": "Unchanged source-validation-selected best.pt",
    }
    protocol_file = args.output / "protocol.json"
    if protocol_file.exists():
        if json.loads(protocol_file.read_text(encoding="utf-8")) != json.loads(json.dumps(protocol)):
            raise ValueError("Protocol differs from the existing experiment")
    else:
        protocol_file.write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    results = []
    for name, size, augment in configurations:
        for dataset in ("bijie", "cas_moxi"):
            data = args.external / dataset / (dataset + ".yaml")
            for key in ("yolo11n", "sf"):
                for seed in (0, 1, 2):
                    identifier = "{}_{}_{}_seed{}".format(name, dataset, key, seed)
                    folder = args.output / identifier
                    summary = folder / "metrics.json"
                    if summary.exists():
                        results.append(json.loads(summary.read_text(encoding="utf-8")))
                        continue
                    if folder.exists():
                        raise RuntimeError("Inspect partial evaluation before retrying: " + str(folder))
                    weights = args.runs / "{}_seed{}".format(key, seed) / "weights/best.pt"
                    model = YOLO(str(weights), task="detect")
                    metrics = model.val(
                        data=str(data), split="test", project=str(args.output), name=identifier,
                        exist_ok=False, imgsz=size, batch=4, workers=0, device=0, half=False,
                        conf=.001, iou=.7, max_det=300, augment=augment,
                        save_json=True, plots=False,
                    )
                    row = dict(configuration=name, dataset=dataset, model=key, seed=seed,
                               map50=float(metrics.box.map50), map5095=float(metrics.box.map),
                               precision=float(metrics.box.mp), recall=float(metrics.box.mr),
                               inference_ms=float(metrics.speed["inference"]))
                    summary.write_text(json.dumps(row, indent=2), encoding="utf-8")
                    results.append(row)
                    print(json.dumps(row), flush=True)
    with (args.output / "per_seed_metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--runs", required=True, type=Path)
    parser.add_argument("--external", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    for key in ("repo", "runs", "external"):
        setattr(args, key, getattr(args, key).resolve(strict=True))
    args.output = args.output.resolve()
    run(args)
