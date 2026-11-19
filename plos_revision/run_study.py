"""Train or evaluate paired-seed detection models using the local YOLO fork."""

import argparse
import json
import os
import sys
from pathlib import Path


SF_ABLATIONS = ("sf_full_qkv", "sf_wide_ffn", "sf_no_eca", "sf_fixed_weights")
SF_VARIANTS = (*SF_ABLATIONS, "sf_qkv16")


def change_sf(model, variant):
    from torch import nn

    block, = (module for module in model.modules()
              if type(module).__name__ == "CNNTransformerParallel_V2_SharedFormer")
    if variant in ("sf_full_qkv", "sf_qkv16"):
        attention = block.attn
        dim = attention.dim
        head_dim = dim // attention.num_heads if variant == "sf_full_qkv" else 16
        projection_dim = head_dim * attention.num_heads
        attention.inner_dim = 3 * projection_dim
        attention.inner_per_head = head_dim
        attention.scale = attention.inner_per_head ** -.5
        attention.qkv_shared = nn.Linear(dim, attention.inner_dim, bias=False)
        attention.out_proj = nn.Linear(projection_dim, dim, bias=False)
    elif variant == "sf_wide_ffn":
        dim = block.ffn[0].in_features
        hidden = int(dim * 1.5)
        block.ffn[0] = nn.Linear(dim, hidden)
        block.ffn[2] = nn.Linear(hidden, dim)
    elif variant == "sf_no_eca":
        block.cnn_eca = block.trans_eca = block.fuse_eca = nn.Identity()
    elif variant == "sf_fixed_weights":
        for name in ("cnn_weight", "trans_weight"):
            value = getattr(block, name).detach().clone()
            delattr(block, name)
            block.register_buffer(name, value)


def run(args):
    os.environ.setdefault("YOLO_CONFIG_DIR", str(args.output / "ultralytics_settings"))
    os.environ.setdefault("YOLO_AUTOINSTALL", "false")
    sys.path.insert(0, str(args.repo))
    from ultralytics import YOLO
    from ultralytics.models.yolo.detect import DetectionTrainer

    configs = args.repo / "ultralytics" / "cfg" / "models" / "11"
    models = {
        "yolo11n": configs / "yolo11n.yaml",  # Ultralytics resolves this to yolo11.yaml.
        "ctpa": configs / "yolo11-cnn-former.yaml",
        "sf": configs / "Att_yaml" / "01" / "08_SharedFormer.yaml",
    }
    models.update({key: models["sf"] for key in SF_VARIANTS})
    for seed in args.seeds:
        for key in args.models:
            name = "{}_seed{}".format(key, seed)
            run_dir = args.output / name
            best = run_dir / "weights" / "best.pt"
            complete = run_dir / "completed.json"
            if args.stage == "train":
                if complete.is_file() and best.is_file():
                    continue
                if run_dir.exists():
                    raise RuntimeError("Partial run exists. Inspect before resuming: " + str(run_dir))
                model = YOLO(str(models[key]), task="detect")
                model.load(str(args.repo / "weights" / "yolo11n.pt"))
                train_options = {}
                if key in SF_VARIANTS:
                    class SFVariantTrainer(DetectionTrainer):
                        def get_model(self, cfg=None, weights=None, verbose=True):
                            detector = super().get_model(cfg=cfg, weights=weights, verbose=verbose)
                            change_sf(detector, key)
                            return detector

                    train_options["trainer"] = SFVariantTrainer
                model.train(
                    data=str(args.data), project=str(args.output), name=name, exist_ok=False,
                    epochs=args.epochs, patience=args.patience, batch=args.batch, imgsz=640,
                    workers=0, device=0, optimizer="AdamW", lr0=.001, cos_lr=True,
                    lrf=.01, weight_decay=.0005, seed=seed, deterministic=True, amp=True,
                    degrees=0, shear=0, perspective=0, translate=0, scale=0, mosaic=0,
                    mixup=0, flipud=0, fliplr=.5, hsv_h=.015, hsv_s=.7, hsv_v=.4,
                    plots=True, **train_options,
                )
                if not best.is_file():
                    raise FileNotFoundError(best)
                complete.write_text(json.dumps({"model": key, "seed": seed}), encoding="utf-8")
            else:
                if not (complete.is_file() and best.is_file()):
                    raise RuntimeError("Missing completed training run: " + str(run_dir))
                model = YOLO(str(best), task="detect")
                parameters = sum(p.numel() for p in model.model.parameters())
                metrics = model.val(
                    data=str(args.data), split=args.split, project=str(args.output / "evaluation"),
                    name="{}_{}_{}".format(args.data.stem, args.split, name), exist_ok=True,
                    batch=args.batch, imgsz=640, workers=0, device=0, half=False,
                    conf=.001, iou=.7, max_det=300, plots=True, save_json=True,
                )
                result = dict(model=key, seed=seed, split=args.split, data=args.data.stem,
                              map50=float(metrics.box.map50), map5095=float(metrics.box.map),
                              precision=float(metrics.box.mp), recall=float(metrics.box.mr),
                              parameters=parameters)
                path = Path(metrics.save_dir) / "metrics.json"
                path.write_text(json.dumps(result, indent=2), encoding="utf-8")
                for x, y, xlabel, ylabel in metrics.curves_results:
                    if xlabel == "Recall" and ylabel == "Precision":
                        curve = dict(recall=x.tolist(), precision=y[0].tolist())
                        (path.parent / "pr_curve.json").write_text(json.dumps(curve), encoding="utf-8")
                        break
                print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=("train", "evaluate"), default="train")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--models", nargs="+", default=["yolo11n", "ctpa", "sf"],
                        choices=("yolo11n", "ctpa", "sf", *SF_VARIANTS))
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--batch", type=int, default=8)
    options = parser.parse_args()
    options.repo = options.repo.resolve(strict=True)
    options.data = options.data.resolve(strict=True)
    options.output = options.output.resolve()
    run(options)
