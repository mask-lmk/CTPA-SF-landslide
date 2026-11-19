"""Freeze a stratified Bijie detection split for independent reproducibility."""

import argparse
import csv
import json
import random
import shutil
from collections import Counter
from pathlib import Path


def prepare(source, destination, seed):
    if destination.exists():
        raise FileExistsError(destination)
    with (source / "manifest.csv").open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    classes = ["landslide", "non-landslide"]
    rng = random.Random(seed)
    assignment = {}
    for category in classes:
        group = sorted(r["id"] for r in rows if r["source"] == category)
        rng.shuffle(group)
        train, val = round(.7 * len(group)), round(.2 * len(group))
        for split, ids in zip(("train", "val", "test"), (group[:train], group[train:train+val], group[train+val:])):
            assignment.update(dict.fromkeys(ids, split))
    work = destination.with_name(destination.name + ".building")
    work.mkdir(parents=True)
    totals = {split: Counter() for split in ("train", "val", "test")}
    for split in totals:
        for kind in ("images", "labels"):
            (work / kind / split).mkdir(parents=True)
    for row in rows:
        split = assignment[row["id"]]
        row["split"] = split
        totals[split].update(images=1, instances=int(row["instances"]))
        for kind, extension in (("images", ".png"), ("labels", ".txt")):
            name = row["id"] + extension
            shutil.copyfile(source / kind / "test" / name, work / kind / split / name)
    with (work / "manifest.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (work / "summary.json").write_text(json.dumps({"split_seed": seed, "splits": totals}, indent=2), encoding="utf-8")
    path = json.dumps(destination.resolve().as_posix(), ensure_ascii=False)
    (work / "bijie_benchmark.yaml").write_text(
        "path: {}\ntrain: images/train\nval: images/val\ntest: images/test\nnames: [landslide]\n".format(path),
        encoding="utf-8"
    )
    work.rename(destination)
    print(json.dumps(totals, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bijie", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", default=42, type=int)
    args = parser.parse_args()
    prepare(args.bijie, args.output, args.seed)
