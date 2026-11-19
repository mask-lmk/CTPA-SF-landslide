"""Recover the audited MLD cohort and split by parent tile before augmentation."""

import argparse
import csv
import hashlib
import json
import random
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image


def prepare(originals, legacy_labels, output, seed):
    if output.exists():
        raise FileExistsError(output)
    sample_ids = sorted({
        re.sub(r"_aug\d+$", "", file.stem)
        for file in legacy_labels.glob("*/*.txt")
        if re.fullmatch(r"mt_\d+_\d+_aug\d+", file.stem)
    })
    if not sample_ids:
        raise ValueError("No MLD source IDs could be traced from the old split.")

    parents = defaultdict(list)
    hash_parents = {}
    for sample in sample_ids:
        image = originals / "mg" / (sample + ".jpg")
        label = originals / "lb" / (sample + ".txt")
        parent = sample.rsplit("_", 1)[0]
        digest = hashlib.sha256(image.read_bytes()).hexdigest()
        if hash_parents.setdefault(digest, parent) != parent:
            raise ValueError("Identical image bytes across parent tiles: " + sample)
        with Image.open(image) as opened:
            width, height = opened.size
            opened.verify()
        boxes = [line.split() for line in label.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
        for box in boxes:
            if len(box) != 5 or float(box[0]) != 0:
                raise ValueError("Invalid YOLO annotation: " + str(label))
            cx, cy, w, h = map(float, box[1:])
            if not (0 < w <= 1 and 0 < h <= 1 and 0 <= cx <= 1 and 0 <= cy <= 1):
                raise ValueError("Invalid YOLO coordinates: " + str(label))
        parents[parent].append((image, label, digest, width, height, len(boxes)))

    keys = sorted(parents)
    if len(keys) < 10:
        raise ValueError("Insufficient independent parent groups for this split.")
    random.Random(seed).shuffle(keys)
    n_train, n_val = round(len(keys) * .7), round(len(keys) * .2)
    assignment = {
        "train": keys[:n_train],
        "val": keys[n_train:n_train + n_val],
        "test": keys[n_train + n_val:],
    }
    work = output.with_name(output.name + ".building")
    work.mkdir(parents=True)
    report, rows = {}, []
    for split, group_ids in assignment.items():
        images, labels = work / "images" / split, work / "labels" / split
        images.mkdir(parents=True)
        labels.mkdir(parents=True)
        counts = Counter()
        sizes = Counter()
        for parent in sorted(group_ids):
            for image, label, digest, width, height, count in parents[parent]:
                shutil.copyfile(image, images / image.name)
                shutil.copyfile(label, labels / label.name)
                rows.append((split, parent, image.name, digest, width, height, count))
                counts.update(images=1, instances=count)
                sizes["{}x{}".format(width, height)] += 1
        report[split] = dict(counts, parents=len(group_ids), sizes=dict(sizes))

    with (work / "manifest.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(("split", "parent_tile", "filename", "sha256", "width", "height", "instances"))
        writer.writerows(rows)
    (work / "summary.json").write_text(json.dumps({"split_seed": seed, "splits": report}, indent=2),
                                       encoding="utf-8")
    (work / "mld.yaml").write_text(
        "path: {}\ntrain: images/train\nval: images/val\ntest: images/test\nnames: [landslide]\n"
        .format(json.dumps(output.resolve().as_posix(), ensure_ascii=False)), encoding="utf-8"
    )
    work.rename(output)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--originals", required=True, type=Path)
    parser.add_argument("--legacy-labels", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    prepare(args.originals, args.legacy_labels, args.output, args.seed)
