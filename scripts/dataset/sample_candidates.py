"""Deterministic group-aware stratified sampling for candidates.

The sampling unit is a group (scenario/family/trajectory): selecting one row
selects the whole group, so counterfactual branches never cross splits.

Strata are primary source x label (and optionally facet). Allocation is
proportional to stratum size with a minimum per stratum; groups are ordered by
a salted SHA-256 of the group key. The achieved counts are reported in the
manifest, because whole-group selection can overshoot the requested total.

Example:
    python scripts/data/sample_candidates.py \
        --input candidates.jsonl \
        --output sample.jsonl \
        --manifest sample.manifest.json \
        --total 15000 --min-per-stratum 50 --seed data-pilot
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


def digest(value):
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


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


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="candidate JSONL")
    parser.add_argument("--output", type=Path, required=True, help="sampled JSONL")
    parser.add_argument("--manifest", type=Path, required=True, help="sampling manifest JSON")
    parser.add_argument("--total", type=int, default=0, help="target rows; 0 means all eligible")
    parser.add_argument("--min-per-stratum", type=int, default=1)
    parser.add_argument("--seed", default="data-sampling")
    parser.add_argument("--key-field", default="key")
    parser.add_argument("--group-field", default="group")
    parser.add_argument("--source-field", default="sources")
    parser.add_argument("--label-field", default="label")
    parser.add_argument("--facet-field", default="facets")
    parser.add_argument("--include-null", action="store_true", help="sample rows with label=null")
    args = parser.parse_args()

    rows = read_rows(args.input)
    for row in rows:
        if args.key_field not in row:
            raise SystemExit(f"missing field {args.key_field!r} in row")
        if args.group_field not in row:
            raise SystemExit(f"missing field {args.group_field!r} in row")

    excluded_null = 0
    if not args.include_null:
        kept = []
        for row in rows:
            if label_name(row.get(args.label_field)) == "null":
                excluded_null += 1
            else:
                kept.append(row)
        rows = kept

    groups = defaultdict(list)
    for row in rows:
        groups[str(row[args.group_field])].append(row)

    def stratum(group_rows):
        labels = sorted({label_name(r.get(args.label_field)) for r in group_rows})
        label = labels[0] if len(labels) == 1 else "mixed"
        sources = sorted({s for r in group_rows for s in as_list(r.get(args.source_field))})
        source = "+".join(sources) if sources else "unknown"
        facets = sorted({f for r in group_rows for f in as_list(r.get(args.facet_field))})
        facet = "+".join(facets) if facets else "unlabelled"
        return source, label, facet

    by_stratum = defaultdict(list)
    for group, group_rows in groups.items():
        by_stratum[stratum(group_rows)].append(group)

    total_rows = len(rows)
    target = total_rows if args.total <= 0 else min(args.total, total_rows)
    selected = set()
    for key, group_keys in sorted(by_stratum.items()):
        desired = max(
            args.min_per_stratum,
            round(target * sum(len(groups[g]) for g in group_keys) / total_rows),
        )
        chosen_rows = 0
        for group in sorted(group_keys, key=lambda g: digest([args.seed, g])):
            if chosen_rows >= desired:
                break
            selected.add(group)
            chosen_rows += len(groups[group])

    selected_rows = [row for row in rows if str(row[args.group_field]) in selected]
    selected_groups = sorted(selected)

    source_counts = Counter()
    label_counts = Counter()
    facet_counts = Counter()
    for row in selected_rows:
        label_counts[label_name(row.get(args.label_field))] += 1
        for source in as_list(row.get(args.source_field)) or ["unknown"]:
            source_counts[source] += 1
        for facet in as_list(row.get(args.facet_field)) or ["unlabelled"]:
            facet_counts[facet] += 1

    write_jsonl(args.output, selected_rows)
    manifest = {
        "input": str(args.input),
        "input_sha256": file_hash(args.input),
        "output": str(args.output),
        "output_sha256": file_hash(args.output),
        "seed": args.seed,
        "target_rows": target,
        "achieved_rows": len(selected_rows),
        "eligible_rows": total_rows,
        "excluded_null": excluded_null,
        "groups_total": len(groups),
        "groups_selected": len(selected_groups),
        "selected_groups_sha256": digest(selected_groups),
        "sources": dict(sorted(source_counts.items())),
        "labels": dict(sorted(label_counts.items())),
        "facets": dict(sorted(facet_counts.items())),
        "strata": {
            "|".join(key): {
                "groups": len(group_keys),
                "rows": sum(len(groups[g]) for g in group_keys),
                "selected_groups": sum(1 for g in group_keys if g in selected),
                "selected_rows": sum(len(groups[g]) for g in group_keys if g in selected),
            }
            for key, group_keys in sorted(by_stratum.items())
        },
        "fields": {
            "key": args.key_field,
            "group": args.group_field,
            "source": args.source_field,
            "label": args.label_field,
            "facet": args.facet_field,
        },
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "rows": len(selected_rows), "groups": len(selected_groups)}))


if __name__ == "__main__":
    main()
