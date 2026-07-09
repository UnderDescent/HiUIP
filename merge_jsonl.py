#!/usr/bin/env python3
"""Merge JSONL files in order."""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("inputs", nargs="+", type=Path)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with args.output.open("w", encoding="utf-8") as out:
        for path in args.inputs:
            with path.open(encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        out.write(line if line.endswith("\n") else line + "\n")
                        count += 1
    print(f"Wrote {count} rows to {args.output}")


if __name__ == "__main__":
    main()
