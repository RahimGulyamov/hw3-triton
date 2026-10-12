"""Fetch pinned pretrained weights at image build time; runtime needs no network."""

import json
from pathlib import Path

import torch
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small


def main():
    torch.set_num_threads(1)
    weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
    model = mobilenet_v3_small(weights=weights).eval()
    scripted = torch.jit.script(model)
    with torch.inference_mode():
        for batch_size in (1, 4, 8):
            sample = torch.zeros(batch_size, 3, 224, 224)
            torch.testing.assert_close(scripted(sample), model(sample))
    target = Path("/models/mobilenet/1")
    scripted.save(str(target / "weights.pt"))
    (target / "labels.json").write_text(json.dumps(weights.meta["categories"], ensure_ascii=False))
    print(f"Exported pretrained MobileNetV3 Small to {target}")


if __name__ == "__main__":
    main()
