"""Measure THOP MACs x 2, parameter count and model-only inference FPS."""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
import torch
from thop import profile
from csds_net.model import build_model, MODEL_NAMES


def measure_complexity(name, image_size, device, output=None, repeats=30, warmup=5):
    if repeats < 1 or warmup < 0:
        raise ValueError("repeats must be positive and warmup nonnegative")
    model = build_model(name).to(device).eval()
    dummy = torch.randn(1, 3, image_size, image_size, device=device)
    params = sum(p.numel() for p in model.parameters()) / 1e6
    with torch.no_grad():
        macs, _ = profile(model, inputs=(dummy,), verbose=False)
        for _ in range(warmup):
            model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        for _ in range(repeats):
            model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize()
    result = {"params_m": params, "gflops": 2 * macs / 1e9,
              "fps": repeats / (time.perf_counter() - started), "device": str(device),
              "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
              "torch": torch.__version__, "input": [1, 3, image_size, image_size],
              "precision": "float32", "warmup": warmup, "repeats": repeats,
              "scope": "model forwards only; excludes loading, preprocessing, metric calculation and saving",
              "method": "THOP counted MACs x 2; operator coverage is profiler-dependent"}
    if output:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps({name:result}, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODEL_NAMES, default="csds_semantic")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", default="runs/profile.json")
    parser.add_argument("--repeats", type=int, default=30)
    args = parser.parse_args()
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
    print(json.dumps(measure_complexity(args.model, args.image_size, device, args.output, args.repeats), indent=2))
