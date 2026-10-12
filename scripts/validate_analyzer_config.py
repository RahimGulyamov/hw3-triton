#!/usr/bin/env python3
"""Parse the profiling YAML with the installed Model Analyzer, without profiling."""

import argparse
import logging
import sys
from importlib.metadata import version
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "config", nargs="?", default="/workspace/profiling/model_analyzer.yaml"
    )
    args = parser.parse_args()
    config_path = Path(args.config).resolve(strict=True)
    installed_version = version("triton-model-analyzer")
    if installed_version != "1.35.0":
        parser.error(f"Expected Model Analyzer 1.35.0, found {installed_version}")

    # Import the same config class used by the installed CLI, rather than a
    # separate copy of the schema that can drift from the pinned dependency.
    from model_analyzer.cli.cli import CLI, ConfigCommandProfile

    logging.basicConfig(level=logging.INFO)
    cli = CLI()
    cli.add_subcommand("profile", "Validate profiling configuration", ConfigCommandProfile())
    original_argv = sys.argv
    try:
        sys.argv = ["model-analyzer", "profile", "-f", str(config_path)]
        cli.parse()
    finally:
        sys.argv = original_argv
    print(f"Model Analyzer {installed_version}: configuration valid: {config_path}")


if __name__ == "__main__":
    main()
