"""Save a single-image probability map and binary mask at checkpoint resolution."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from csds_net.checkpoint import forward_finite, load_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output-dir", default="runs/prediction")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", action="store_true", help="Default is FP32")
    args = parser.parse_args()
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
    model, config, epoch = load_model(args.checkpoint, device)
    size = config.get("image_size", 256)
    image = Image.open(args.image).convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
    tensor = torch.from_numpy(np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255)[None].to(device)
    output, retries = forward_finite(model, tensor, args.amp and device.type == "cuda")
    probability = output["logits"].sigmoid()[0, 0].cpu().numpy()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    image.save(out / "input.png")
    np.save(out / "probability.npy", probability)
    Image.fromarray((probability * 255).round().astype(np.uint8)).save(out / "probability.png")
    Image.fromarray(((probability >= 0.5) * 255).astype(np.uint8)).save(out / "mask.png")
    (out / "prediction.json").write_text(json.dumps({"checkpoint_epoch": epoch, "threshold": 0.5,
        "output_size": [size, size], "fp32_retry_batches": retries}, indent=2) + "\n", encoding="utf-8")
    print(out.resolve())


if __name__ == "__main__":
    main()
