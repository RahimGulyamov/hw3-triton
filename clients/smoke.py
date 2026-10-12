"""Check HTTP, gRPC, client batching, malformed input, and recovery."""

import argparse

import numpy as np
import torch
from tritonclient.utils import InferenceServerException

from infer import infer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-batch-size", type=int, choices=(1, 4, 8), default=8)
    args = parser.parse_args()
    torch.set_num_threads(1)
    model = torch.jit.load("/models/mobilenet/1/weights.pt").eval()
    rng = np.random.default_rng(123)
    for protocol, url in (("http", "triton:8000"), ("grpc", "triton:8001")):
        for batch in (1, 4, 8):
            if batch > args.max_batch_size:
                continue
            images = rng.normal(size=(batch, 3, 224, 224)).astype(np.float32)
            actual = infer(protocol, url, images)
            with torch.inference_mode():
                expected = model(torch.from_numpy(images)).numpy()
            np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-4)
            print(f"PASS {protocol} batch={batch}: remote/local logits match")
        try:
            infer(protocol, url, np.zeros((1, 3, 100, 100), dtype=np.float32))
        except InferenceServerException:
            print(f"PASS {protocol}: malformed shape rejected")
        else:
            raise AssertionError("Malformed shape unexpectedly accepted")
        valid = infer(protocol, url, np.zeros((1, 3, 224, 224), dtype=np.float32))
        assert valid.shape == (1, 1000) and np.isfinite(valid).all()
        print(f"PASS {protocol}: inference still works after invalid request")


if __name__ == "__main__":
    main()
