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
    if (directory / "model_analyzer/results/metrics-model-inference.csv").is_file():
        from model_analyzer_report import read_measurements
        measurements = read_measurements(directory)
        curves = [
            ("mobilenet_config_default", "Baseline: 1 instance, no batching"),
            ("mobilenet_config_4", "2 instances, max batch 4, delay 0"),
            ("mobilenet_config_5", "2 instances, max batch 8, delay 0"),
            ("mobilenet_config_11", "2 instances, max batch 8, delay 1 ms"),
        ]
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
        fig.suptitle("Model Analyzer: baseline and three highest-throughput configurations")
        for name, label in curves:
            rows = sorted((row for row in measurements if row["name"] == name),
                          key=lambda row: row["concurrency"])
            if not rows:
                raise ValueError(f"Missing measurements for {name}")
            x = [row["concurrency"] for row in rows]
            axes[0].plot(x, [row["throughput"] for row in rows], "o-", label=label)
            axes[1].plot(x, [row["p99_ms"] for row in rows], "o-", label=label)
        axes[0].set(ylabel="Throughput (infer/s)", ylim=(0, None))
        axes[1].set(ylabel="p99 latency (ms)", ylim=(0, None))
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside lower center", ncols=2, fontsize=9)
        for ax in axes:
            ax.set_xscale("log", base=2)
            ax.set_xticks([1, 2, 4, 8, 16], labels=[1, 2, 4, 8, 16])
            ax.set_xlabel("Client concurrency")
            ax.grid(alpha=0.25)
        for suffix in ("png", "svg"):
            fig.savefig(figures / f"model_analyzer.{suffix}", dpi=160)
        plt.close(fig)
    print(f"Figures saved to {figures}")


if __name__ == "__main__":
    main()
