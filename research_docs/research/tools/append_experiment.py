#!/usr/bin/env python3
"""Append one JSON experiment record to research/EXPERIMENTS.jsonl."""

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("record", help="Path to a JSON experiment record")
    parser.add_argument("--ledger", default="research/EXPERIMENTS.jsonl")
    args = parser.parse_args()

    record_path = Path(args.record)
    ledger_path = Path(args.ledger)
    record = json.loads(record_path.read_text())

    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    with ledger_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    print(f"appended {record.get('id', '<no-id>')} -> {ledger_path}")


if __name__ == "__main__":
    main()
