"""Render measured Performance Analyzer logs as Markdown (standard library only)."""

import argparse
from pathlib import Path
import re


def last_number(text, pattern):
    values = re.findall(pattern, text, flags=re.IGNORECASE)
    if not values:
        raise ValueError(f"Measurement not found: {pattern}")
    return float(values[-1])


def render(directory):
    paths = sorted(directory.glob("concurrency_*.log"), key=lambda p: int(p.stem.split("_")[-1]))
    if not paths:
        raise ValueError(f"No measurement logs in {directory}; run Performance Analyzer first")
    lines = ["| Concurrency | Throughput, infer/s | Mean latency, ms | p99 latency, ms |", "| ---: | ---: | ---: | ---: |"]
    for path in paths:
        body = path.read_text()
        concurrency = int(path.stem.split("_")[-1])
        throughput = last_number(body, r"Throughput:\s*([\d.eE+\-]+)\s*infer/sec")
        average = last_number(body, r"Avg latency:\s*([\d.eE+\-]+)\s*usec") / 1000
        p99 = last_number(body, r"p99 latency:\s*([\d.eE+\-]+)\s*usec") / 1000
        lines.append(f"| {concurrency} | {throughput:.3f} | {average:.3f} | {p99:.3f} |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    print(render(args.directory))


if __name__ == "__main__":
    main()
