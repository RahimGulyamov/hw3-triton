"""Plot actual Performance Analyzer logs and custom metrics from a results folder."""

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from perf_report import last_number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    directory = args.directory
    figures = directory / "figures"
    figures.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 11, "svg.fonttype": "none"})

    paths = sorted((directory / "perf_analyzer/baseline").glob("concurrency_*.log"),
                   key=lambda p: int(p.stem.split("_")[-1]))
    if not paths:
        raise ValueError("No Performance Analyzer logs")
    concurrency, throughput, mean, p99 = [], [], [], []
    for path in paths:
        text = path.read_text()
        concurrency.append(int(path.stem.split("_")[-1]))
        throughput.append(last_number(text, r"Throughput:\s*([\d.eE+\-]+)\s*infer/sec"))
        mean.append(last_number(text, r"Avg latency:\s*([\d.eE+\-]+)\s*usec") / 1000)
        p99.append(last_number(text, r"p99 latency:\s*([\d.eE+\-]+)\s*usec") / 1000)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), layout="constrained")
    fig.suptitle("MobileNetV3 Small: one CPU instance, no dynamic batching")
    axes[0].plot(concurrency, throughput, "o-", color="#2563eb")
    axes[0].set(ylabel="Throughput (infer/s)", ylim=(0, max(throughput) * 1.15))
    axes[1].plot(concurrency, mean, "o-", label="Mean", color="#2563eb")
    axes[1].plot(concurrency, p99, "s-", label="p99", color="#ea580c")
    axes[1].set(ylabel="Client latency (ms)", ylim=(0, max(p99) * 1.15))
    axes[1].legend()
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xticks(concurrency, labels=concurrency)
        ax.set_xlabel("Client concurrency")
        ax.grid(alpha=0.25)
    for suffix in ("png", "svg"):
        fig.savefig(figures / f"performance.{suffix}", dpi=160)
    plt.close(fig)

    with (directory / "metrics/samples.csv").open() as file:
        samples = list(csv.DictReader(file))
    origin = float(samples[0]["unix_time"])
    seconds = [float(row["unix_time"]) - origin for row in samples]
    counter = [float(row["hw3_processing_seconds_total"]) for row in samples]
    gauge = [float(row["hw3_inflight_requests"]) for row in samples]
    fig, axes = plt.subplots(2, 1, figsize=(9, 5.5), sharex=True, layout="constrained")
    fig.suptitle("Custom metrics: 320 requests, 16 clients, one CPU instance")
    axes[0].plot(seconds, counter, color="#2563eb")
    axes[0].set_ylabel("Processing COUNTER (s)")
    axes[1].step(seconds, gauge, where="post", color="#059669")
    axes[1].set(ylabel="In-flight GAUGE (requests)", xlabel="Time since first sample (s)",
                ylim=(-0.1, max(gauge) + 0.2))
    axes[1].set_yticks(sorted(set(gauge)))
    for ax in axes:
        ax.grid(alpha=0.25)
    for suffix in ("png", "svg"):
        fig.savefig(figures / f"metrics.{suffix}", dpi=160)
    plt.close(fig)
    print(f"Figures saved to {figures}")


if __name__ == "__main__":
    main()
