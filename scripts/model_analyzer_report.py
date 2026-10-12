"""Render actual Model Analyzer CSV and saved model configurations as Markdown."""

import argparse
import csv
from pathlib import Path
import re


def configuration(directory, name):
    text = (directory / "configurations" / name / "config.pbtxt").read_text()
    def number(field, default=None):
        match = re.search(rf"\b{field}\s*:\s*(\d+)", text)
        if match:
            return int(match[1])
        if default is None:
            raise ValueError(f"Missing {field} in {name}")
        return default
    return {
        "max_batch_size": number("max_batch_size"),
        "instances": number("count"),
        "dynamic_batching": bool(re.search(r"\bdynamic_batching\s*\{", text)),
        "queue_delay_us": number("max_queue_delay_microseconds", 0),
    }


def read_measurements(directory):
    path = directory / "model_analyzer/results/metrics-model-inference.csv"
    measurements = []
    seen = set()
    with path.open(newline="") as file:
        for row in csv.DictReader(file):
            name = row["Model Config Path"]
            config = configuration(directory, name)
            concurrency = int(row["Concurrency"])
            key = (name, int(row["Batch"]), concurrency)
            if key in seen:
                raise ValueError(f"Duplicate measurement: {key}")
            seen.add(key)
            if int(row["Max Batch Size"]) != config["max_batch_size"]:
                raise ValueError(f"Config/CSV max batch size mismatch: {name}")
            if row["Instance Group"] != f"{config['instances']}:CPU":
                raise ValueError(f"Config/CSV instance group mismatch: {name}")
            measurements.append({
                "name": name, **config,
                "concurrency": concurrency,
                "throughput": float(row["Throughput (infer/sec)"]),
                "mean_ms": float(row["Avg Latency (ms)"]),
                "p99_ms": float(row["p99 Latency (ms)"]),
            })
    if not measurements:
        raise ValueError(f"No measurements in {path}")
    return measurements


def render(directory, concurrency=None):
    measurements = read_measurements(directory)
    by_name = {}
    for measurement in measurements:
        if concurrency is not None and measurement["concurrency"] != concurrency:
            continue
        name = measurement["name"]
        if name not in by_name or measurement["throughput"] > by_name[name]["throughput"]:
            by_name[name] = measurement
    if not by_name:
        raise ValueError(f"No measurements for concurrency={concurrency}")
    lines = [
        "| Config | Instances | Max batch | Queue delay, µs | Concurrency | Throughput, infer/s | Mean, ms | p99, ms |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in sorted(by_name.values(), key=lambda x: x["throughput"], reverse=True):
        delay = str(row["queue_delay_us"]) if row["dynamic_batching"] else "disabled"
        lines.append(
            f"| {row['name']} | {row['instances']} | {row['max_batch_size']} | {delay} | "
            f"{row['concurrency']} | {row['throughput']:.1f} | {row['mean_ms']:.1f} | {row['p99_ms']:.1f} |"
        )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--concurrency", type=int)
    args = parser.parse_args()
    print(render(args.directory, args.concurrency))


if __name__ == "__main__":
    main()
