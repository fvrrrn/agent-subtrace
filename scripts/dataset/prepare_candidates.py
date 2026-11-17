"""Normalize candidate JSONL for sampling and annotation.

Raw candidates use `provenance` (list of dicts) rather than a flat `sources`
field, and `state` may be a dict or a JSON string. This script adds:
  - sources: sorted unique provenance.source
  - facets: [] (not annotated yet)
  - state: parsed dict
and drops nothing else.

Example:
    python prepare_candidates.py \
        --input actions.jsonl \
        --output candidates.normalized.jsonl
"""
import argparse
import json
from pathlib import Path


def sources_of(row):
    result = set()
    for item in row.get("provenance") or []:
        if isinstance(item, dict) and item.get("source"):
            result.add(str(item["source"]))
    if not result and row.get("sources"):
        result.update(str(s) for s in row["sources"])
    return sorted(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    seen = 0
    with args.input.open("r", encoding="utf-8") as source, args.output.open("w", encoding="utf-8") as sink:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise SystemExit(f"{args.input}:{line_number}: {error}") from error
            state = row.get("state")
            if isinstance(state, str):
                state = json.loads(state)
            row["state"] = state
            row["sources"] = sources_of(row)
            row.setdefault("facets", [])
            sink.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            seen += 1
    print(json.dumps({"output": str(args.output), "rows": seen}))


if __name__ == "__main__":
    main()
