#!/usr/bin/env python3
"""Keep only direct-child JSONL rows whose site appears in an allowlist."""

import argparse
import json
from pathlib import Path


def load_sites(path):
    sites = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                sites.append(line)
    return set(sites)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sites", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    allowed = load_sites(args.sites)
    kept = 0
    total = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.input.open(encoding="utf-8") as src, args.output.open("w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue
            total += 1
            row = json.loads(line)
            if row.get("site") in allowed:
                dst.write(line if line.endswith("\n") else line + "\n")
                kept += 1

    print(f"Allowed sites: {len(allowed)}")
    print(f"Kept {kept}/{total} rows -> {args.output}")


if __name__ == "__main__":
    main()
