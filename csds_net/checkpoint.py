"""Load trusted training checkpoints, including historical net.* state keys."""
from __future__ import annotations
import hashlib
from pathlib import Path
import torch
from .model import build_model


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_model(path, device="cpu"):
    # Training checkpoints contain Python optimizer/config objects. Load only
    # checkpoints from a trusted source; weights_only=False is intentional.
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = checkpoint.get("config", {})
    name = config.get("model", "csds_semantic")
    state = checkpoint["model"]
    if all(key.startswith("net.") for key in state):
        state = {key[4:]: value for key, value in state.items()}
    model = build_model(name, pretrained=False)
    model.load_state_dict(state, strict=True)
    if not all(torch.isfinite(value).all() for value in model.state_dict().values()):
        raise FloatingPointError("Checkpoint contains non-finite parameters/buffers")
    return model.to(device).eval(), config, checkpoint.get("epoch")


@torch.no_grad()
def forward_finite(model, images, amp=False):
    with torch.autocast(device_type=images.device.type, enabled=amp):
        output = model(images)
    retry = not all(torch.isfinite(value).all() for value in output.values())
    if retry:
        with torch.autocast(device_type=images.device.type, enabled=False):
            output = model(images.float())
    if not all(torch.isfinite(value).all() for value in output.values()):
        raise FloatingPointError("Non-finite model output after FP32 retry")
    return output, int(retry)
