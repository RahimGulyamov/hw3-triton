"""Rename a saved Model Analyzer configuration for the mobilenet directory."""

import argparse
from pathlib import Path
import re


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    content, count = re.subn(r'^name\s*:\s*"[^"]*"', 'name: "mobilenet"', args.config.read_text(), count=1, flags=re.MULTILINE)
    if count != 1:
        raise ValueError("Top-level model name was not found in the config")
    args.config.write_text(content)


if __name__ == "__main__":
    main()
