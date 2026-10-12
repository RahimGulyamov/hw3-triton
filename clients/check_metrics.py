"""Observe custom COUNTER/GAUGE before, during, and after real HTTP load."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import json
from pathlib import Path
import re
import time
from urllib.request import urlopen

import numpy as np
from infer import create_client


def scrape(url):
    with urlopen(url, timeout=5) as response:
        body = response.read().decode()
    values = {}
    for name in ("hw3_processing_seconds_total", "hw3_inflight_requests"):
        samples = re.findall(r"^" + name + r"\{[^\n]*\}\s+([-+\d.eE]+)$", body, re.MULTILINE)
        if not samples:
            raise RuntimeError(f"Missing custom metric: {name}")
        values[name] = sum(float(sample) for sample in samples)
    return values


def worker(url, images, requests):
    module, client = create_client("http", url)
    inp = module.InferInput("IMAGE", images.shape, "FP32")
    inp.set_data_from_numpy(images)
    try:
        for _ in range(requests):
            result = client.infer("mobilenet", [inp]).as_numpy("LOGITS")
            assert result.shape == (1, 1000) and np.isfinite(result).all()
    finally:
        client.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="localhost:8000")
    parser.add_argument("--metrics-url", default="http://localhost:8002/metrics")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--requests-per-worker", type=int, default=20)
    parser.add_argument("--output", type=Path, default=Path("/results/metrics"))
    args = parser.parse_args()
    if args.workers < 1 or args.requests_per_worker < 1:
        parser.error("workers and requests-per-worker must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    images = np.zeros((1, 3, 224, 224), dtype=np.float32)
    initial = scrape(args.metrics_url)
    observations = [(time.time(), initial)]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(worker, args.url, images, args.requests_per_worker) for _ in range(args.workers)]
        while not all(future.done() for future in futures):
            observations.append((time.time(), scrape(args.metrics_url)))
            time.sleep(0.02)
        for future in futures:
            future.result()
    final = scrape(args.metrics_url)
    observations.append((time.time(), final))
    counter = "hw3_processing_seconds_total"
    gauge = "hw3_inflight_requests"
    with (args.output / "samples.csv").open("w", newline="") as target:
        writer = csv.writer(target)
        writer.writerow(["unix_time", counter, gauge])
        writer.writerows((timestamp, sample[counter], sample[gauge]) for timestamp, sample in observations)
    summary = {
        "counter_increase_seconds": final[counter] - initial[counter],
        "gauge_max_observed": max(sample[gauge] for _, sample in observations),
        "gauge_after_load": final[gauge],
        "successful_requests": args.workers * args.requests_per_worker,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    assert summary["counter_increase_seconds"] > 0, "COUNTER did not increase"
    assert summary["gauge_max_observed"] > 0, "GAUGE was not observed; increase load and repeat"
    assert summary["gauge_after_load"] == 0, "GAUGE did not return to zero"


if __name__ == "__main__":
    main()
