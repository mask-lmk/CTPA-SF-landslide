"""Prepare model-independent Bijie and CAS Moxi detection evaluation sets."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


def external_sources(bijie, cas_manifest):
    for category in ("landslide", "non-landslide"):
        root = bijie / category
        for path in sorted((root / "image").iterdir()):
            if path.suffix.lower() in {".jpg", ".png", ".tif", ".tiff"}:
                mask = root / "mask" / (path.stem + ".png") if category == "landslide" else None
                yield "bijie", category, path, mask
    if cas_manifest is None:
        return
    with cas_manifest.open(encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            if row["region_id"] != "Moxi town":
                raise ValueError("Incorrect region in Moxi manifest")
            yield "cas_moxi", row["sensor"], Path(row["image_path"]), Path(row["mask_path"])


def read(path, mode):
    data = cv2.imdecode(np.frombuffer(path.read_bytes(), dtype=np.uint8), mode)
    if data is None:
        raise ValueError("Cannot decode " + str(path))
    return data


def prepare(bijie, cas_manifest, output, min_area):
    if output.exists():
        raise FileExistsError(output)
    work = output.with_name(output.name + ".building")
    work.mkdir(parents=True)
    manifests = {"bijie": []}
    if cas_manifest is not None:
        manifests["cas_moxi"] = []
    seen = {}
    for dataset in manifests:
        for kind in ("images", "labels"):
            (work / dataset / kind / "test").mkdir(parents=True)
    for dataset, source, image_path, mask_path in external_sources(bijie, cas_manifest):
        image = read(image_path, cv2.IMREAD_COLOR)
        h, w = image.shape[:2]
        sid = source.replace("-", "_") + "_" + image_path.stem
        labels = []
        if mask_path is not None:
            mask = read(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask.shape != (h, w):
                raise ValueError("Image/mask dimensions differ: " + str(image_path))
            count, _, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype("uint8"), connectivity=8)
            for x, y, bw, bh, area in stats[1:count]:
                if area >= min_area:
                    labels.append("0 {:.8f} {:.8f} {:.8f} {:.8f}".format(
                        (x + bw / 2) / w, (y + bh / 2) / h, bw / w, bh / h))
        digest = hashlib.sha256(image.tobytes()).hexdigest()
        identity = (image.shape, digest)
        if identity in seen:
            prior_dataset, prior_labels = seen[identity]
            if prior_labels != labels or prior_dataset != dataset:
                raise ValueError("Conflicting labels or provenance for identical image: " + str(image_path))
            continue
        seen[identity] = dataset, labels
        base = work / dataset
        target = base / "images" / "test" / (sid + ".png")
        if target.exists():
            raise ValueError("Duplicate output ID: " + sid)
        encoded, buffer = cv2.imencode(".png", image)
        if not encoded:
            raise ValueError("Cannot encode " + sid)
        target.write_bytes(buffer.tobytes())
        (base / "labels" / "test" / (sid + ".txt")).write_text("\n".join(labels), encoding="utf-8")
        manifests[dataset].append((sid, source, image_path.name, len(labels), w, h, digest))
    for dataset, rows in manifests.items():
        with (work / dataset / "manifest.csv").open("w", encoding="utf-8", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(("id", "source", "original_name", "instances", "width", "height", "sha256_pixels"))
            writer.writerows(rows)
        path = json.dumps((output / dataset).resolve().as_posix(), ensure_ascii=False)
        (work / dataset / (dataset + ".yaml")).write_text(
            "path: {}\ntrain: images/test\nval: images/test\ntest: images/test\nnames: [landslide]\n"
            .format(path), encoding="utf-8"
        )
        print(dataset, len(rows), sum(r[3] for r in rows), flush=True)
    work.rename(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bijie", type=Path, required=True)
    parser.add_argument("--cas-manifest", type=Path,
                        help="Optional CAS Moxi index, needed only for external CAS preparation")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-area", type=int, default=1)
    args = parser.parse_args()
    prepare(args.bijie, args.cas_manifest, args.output, args.min_area)
