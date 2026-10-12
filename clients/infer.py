"""HTTP/gRPC examples, image preprocessing, and comparison to local inference."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torchvision.models import MobileNet_V3_Small_Weights
import tritonclient.grpc as grpc
import tritonclient.http as http


def create_client(protocol, url):
    module = http if protocol == "http" else grpc
    return module, module.InferenceServerClient(url=url)


def infer(protocol, url, images):
    module, client = create_client(protocol, url)
    try:
        inp = module.InferInput("IMAGE", images.shape, "FP32")
        inp.set_data_from_numpy(images)
        return client.infer("mobilenet", [inp], outputs=[module.InferRequestedOutput("LOGITS")]).as_numpy("LOGITS")
    finally:
        client.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", choices=["http", "grpc"], default="http")
    parser.add_argument("--url", help="host:port; default localhost:8000/8001")
    parser.add_argument("--image", type=Path)
    parser.add_argument("--batch-size", type=int, choices=range(1, 9), default=1)
    parser.add_argument("--verify", action="store_true", help="Compare to weights baked into the image")
    args = parser.parse_args()
    url = args.url or ("localhost:8000" if args.protocol == "http" else "localhost:8001")
    if args.image:
        with Image.open(args.image) as image:
            tensor = MobileNet_V3_Small_Weights.IMAGENET1K_V1.transforms()(image.convert("RGB"))
        images = np.repeat(tensor.numpy()[None], args.batch_size, axis=0)
    else:
        images = np.random.default_rng(42).normal(size=(args.batch_size, 3, 224, 224)).astype(np.float32)
    logits = infer(args.protocol, url, images)
    assert logits.shape == (args.batch_size, 1000), logits.shape
    assert np.isfinite(logits).all(), "Nonfinite logits"
    if args.verify:
        torch.set_num_threads(1)
        local = torch.jit.load("/models/mobilenet/1/weights.pt").eval()
        with torch.inference_mode():
            expected = local(torch.from_numpy(images)).numpy()
        np.testing.assert_allclose(logits, expected, rtol=1e-4, atol=1e-4)
        print("Remote/local logits match")
    labels = json.loads(Path("/models/mobilenet/1/labels.json").read_text())
    probabilities = torch.softmax(torch.from_numpy(logits), dim=-1).numpy()
    for index, scores in enumerate(probabilities):
        top = scores.argsort()[-5:][::-1]
        print(json.dumps({"item": index, "top5": [{"label": labels[i], "probability": float(scores[i])} for i in top]}))
    if args.image is None:
        print("Synthetic input verifies the protocol; it does not measure classification accuracy.")


if __name__ == "__main__":
    main()
