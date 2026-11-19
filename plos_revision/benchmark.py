"""Measure batch-one FP32 inference with identical inputs and explicit timing scope."""

import argparse
import json
import sys
from pathlib import Path
from statistics import mean, stdev

import torch


def benchmark(repo, runs, output):
    sys.path.insert(0, str(repo.resolve()))
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import get_flops

    torch.manual_seed(42)
    tensor = torch.rand((1, 3, 640, 640), device="cuda:0")
    reports = []
    for key in ("yolo11n", "ctpa", "sf"):
        model = YOLO(str(runs / (key + "_seed0") / "weights/best.pt")).model.float().eval()
        parameters = sum(p.numel() for p in model.parameters())
        flops = get_flops(model, 640)
        model.to("cuda:0").fuse(verbose=False)
        with torch.inference_mode():
            for _ in range(50):
                model(tensor)
            torch.cuda.synchronize()
            samples = []
            for _ in range(5):
                start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
                start.record()
                for _ in range(100):
                    model(tensor)
                end.record()
                torch.cuda.synchronize()
                samples.append(start.elapsed_time(end) / 100)
        reports.append({"model": key, "parameters": parameters, "gflops": flops,
                        "latency_ms": mean(samples), "latency_sd_ms": stdev(samples),
                        "fps": 1000 / mean(samples)})
    output.write_text(json.dumps({"device": torch.cuda.get_device_name(), "warmup": 50,
                                  "repetitions": 5, "forwards_per_repeat": 100, "batch": 1,
                                  "input": 640, "precision": "FP32", "scope": "network forward and decoding",
                                  "results": reports}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--runs", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    benchmark(args.repo, args.runs, args.output)
