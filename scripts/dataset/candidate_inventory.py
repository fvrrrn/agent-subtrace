"""Inventory a candidate JSONL.

Counts candidates by source, label, facet and group without modifying the input.
Intended for the server-side candidate pool; stdlib only.

Example:
    python scripts/dataset/candidate_inventory.py \
        --input candidates.jsonl \
        --output inventory.json
"""
import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


def as_list(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(v) for v in value]
    return [str(value)]


def label_name(value):
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "1" if value else "0"
    return str(value)


def file_hash(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_rows(path):
    rows = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise SystemExit(f"{path}:{line_number}: {error}") from error
    return rows


def inventory(rows, key_field, group_field, source_field, label_field, facet_field):
    keys = Counter()
    sources = Counter()
    labels = Counter()
    facets = Counter()
    source_label = Counter()
    facet_source = Counter()
    group_sizes = Counter()
    group_labels = defaultdict(set)
    missing = Counter()

    for row in rows:
        key = row.get(key_field)
        if key is None:
            missing[key_field] += 1
        else:
            keys[str(key)] += 1

        group = row.get(group_field)
        if group is None:
            missing[group_field] += 1
            group = "unknown"
        group_sizes[str(group)] += 1
        group_labels[str(group)].add(label_name(row.get(label_field)))

        row_sources = as_list(row.get(source_field)) or ["unknown"]
        row_facets = as_list(row.get(facet_field)) or ["unlabelled"]
        row_label = label_name(row.get(label_field))

        for source in row_sources:
            sources[source] += 1
            source_label[(source, row_label)] += 1
        for facet in row_facets:
            facets[facet] += 1
            for source in row_sources:
                facet_source[(facet, source)] += 1
        labels[row_label] += 1

    duplicate_keys = {k: v for k, v in keys.items() if v > 1}
    mixed_groups = {g: sorted(v) for g, v in group_labels.items() if len(v) > 1}

    return {
        "rows": len(rows),
        "unique_keys": len(keys),
        "duplicate_keys": len(duplicate_keys),
        "duplicate_key_examples": dict(list(sorted(duplicate_keys.items()))[:20]),
        "groups": len(group_sizes),
        "group_size_max": max(group_sizes.values(), default=0),
        "group_size_mean": round(sum(group_sizes.values()) / len(group_sizes), 4) if group_sizes else 0,
        "groups_with_mixed_labels": len(mixed_groups),
        "mixed_label_examples": dict(list(sorted(mixed_groups.items()))[:20]),
        "labels": dict(sorted(labels.items())),
        "sources": dict(sorted(sources.items())),
        "source_label": {f"{s}|{l}": c for (s, l), c in sorted(source_label.items())},
        "facets": dict(sorted(facets.items())),
        "facet_source": {f"{f}|{s}": c for (f, s), c in sorted(facet_source.items())},
        "missing_fields": dict(sorted(missing.items())),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="candidate JSONL")
    parser.add_argument("--output", type=Path, required=True, help="inventory JSON")
    parser.add_argument("--key-field", default="key")
    parser.add_argument("--group-field", default="group")
    parser.add_argument("--source-field", default="sources")
    parser.add_argument("--label-field", default="label")
    parser.add_argument("--facet-field", default="facets")
    args = parser.parse_args()

    rows = read_rows(args.input)
    result = {
        "input": str(args.input),
        "input_sha256": file_hash(args.input),
        "fields": {
            "key": args.key_field,
            "group": args.group_field,
            "source": args.source_field,
            "label": args.label_field,
            "facet": args.facet_field,
        },
        **inventory(
            rows,
            args.key_field,
            args.group_field,
            args.source_field,
            args.label_field,
            args.facet_field,
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "rows": result["rows"], "groups": result["groups"]}))


if __name__ == "__main__":
    main()
