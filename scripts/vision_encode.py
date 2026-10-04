#!/usr/bin/env python3
"""Optional vision frontend; the language forward pass remains native Helios.

Loads only model.visual tensors and emits fp16 [image_tokens, hidden] embeddings.
Requires torch, transformers with GLM5Next, safetensors, numpy and Pillow.
The larger local checkpoint has a dense BF16 tower. EXL3 vision towers are
rejected explicitly until a quantized vision implementation is available.
"""
import argparse
import json
import math
from pathlib import Path


def preprocess(image, settings):
    import numpy as np
    import torch
    import torch.nn.functional as F
    patch = settings.get("patch_size", 14)
    merge = settings.get("merge_size", 2)
    temporal = settings.get("temporal_patch_size", 2)
    factor = patch * merge
    minimum = settings.get("min_image_tokens", 16)
    maximum = settings.get("max_image_tokens", 8000)
    height, width = image.height, image.width
    align = lambda n: math.ceil(n / factor) * factor
    target_h, target_w = align(height), align(width)
    if target_h * target_w < minimum * factor**2:
        scale = math.sqrt(minimum * factor**2 / (height * width))
        target_h, target_w = align(math.ceil(height * scale)), align(math.ceil(width * scale))
    if target_h * target_w > maximum * factor**2:
        low, high = 1, height
        target_h = target_w = factor
        while low <= high:
            h = (low + high) // 2
            w = max(1, math.floor(width * h / height))
            if align(h) * align(w) <= maximum * factor**2:
                target_h, target_w = align(h), align(w)
                low = h + 1
            else:
                high = h - 1
    scale = min(target_h / height, target_w / width)
    if height * width >= minimum * factor**2:
        scale = min(scale, 1.0)
    h, w = max(1, math.floor(height * scale)), max(1, math.floor(width * scale))
    pixels = torch.from_numpy(np.array(image.convert("RGB"))).permute(2, 0, 1)[None]
    if (h, w) != (height, width):
        pixels = F.interpolate(pixels, (h, w), mode="bicubic", align_corners=False, antialias=True)
    pixels = F.pad(pixels, (0, target_w - w, 0, target_h - h)).float() / 255
    mean = torch.tensor(settings.get("image_mean", [0.48145466, 0.4578275, 0.40821073]))[None, :, None, None]
    std = torch.tensor(settings.get("image_std", [0.26862954, 0.26130258, 0.27577711]))[None, :, None, None]
    pixels = (pixels - mean) / std
    gh, gw = target_h // patch, target_w // patch
    pixels = pixels.reshape(1, 3, gh // merge, merge, patch, gw // merge, merge, patch)
    pixels = pixels.permute(0, 2, 5, 3, 6, 1, 4, 7)
    pixels = pixels.unsqueeze(6).expand(-1, -1, -1, -1, -1, -1, temporal, -1, -1)
    return pixels.reshape(gh * gw, 3 * temporal * patch * patch), torch.tensor([[1, gh, gw]])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    import torch
    from PIL import Image
    from safetensors import safe_open
    from transformers.models.glm5_next.configuration_glm5_next import Glm5NextVisionConfig
    from transformers.models.glm5_next.modeling_glm5_next import Glm5NextVisionModel
    root = Path(args.model)
    config = Glm5NextVisionConfig(**json.loads((root / "config.json").read_text())["vision_config"])
    config._attn_implementation = "sdpa"
    settings = json.loads((root / "processor_config.json").read_text())["image_processor"]
    # The rotary frequency buffer is non-persistent and must be initialized;
    # loading a state dict cannot materialize it from a meta-device constructor.
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        tower = Glm5NextVisionModel(config)
    finally:
        torch.set_default_dtype(previous_dtype)
    state = {}
    index = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
    files = sorted({v for k, v in index.items() if k.startswith("model.visual.")})
    for file in files:
        with safe_open(root / file, framework="pt", device="cpu") as shard:
            for key in shard.keys():
                if key.startswith("model.visual."):
                    if key.endswith((".trellis", ".suh", ".svh", ".mul1")):
                        raise RuntimeError("EXL3-quantized vision towers are not supported by this frontend")
                    state[key.removeprefix("model.visual.")] = shard.get_tensor(key)
    tower.load_state_dict(state, strict=True, assign=True)
    tower = tower.to(f"cuda:{args.device}").eval()
    with Image.open(args.image) as image:
        pixels, grid = preprocess(image, settings)
    with torch.inference_mode():
        features = tower(pixels.to(tower.device, dtype=tower.dtype), grid.to(tower.device)).pooler_output
    features = features.to(device="cpu", dtype=torch.float16).contiguous()
    if features.shape[1] != config.out_hidden_size or not torch.isfinite(features).all():
        raise RuntimeError("invalid vision embeddings")
    Path(args.output).write_bytes(features.numpy().tobytes())
    Path(args.output + ".json").write_text(json.dumps({"rows": features.shape[0], "hidden": features.shape[1]}))


if __name__ == "__main__":
    main()
