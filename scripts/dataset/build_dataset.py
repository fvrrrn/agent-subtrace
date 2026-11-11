"""Build the action-level safety dataset from silver annotations.

Joins `annotations_v3a_full.jsonl` with the candidate pool, drops gold cases and
AgentDoG rows, assigns group-disjoint splits, samples per the quotas in
`docs/DATA_SPEC.md`, and writes `train/dev/test/ood` jsonl plus a manifest.

Two passes over the pool: the first keeps only small per-row metadata (the
1.1 GB pool never has to be held in memory), the second streams the pool and
emits only the selected rows with their `state` as `input`.

    python3 build_dataset.py \
        --annotations annotations_v3a_full.jsonl \
        --candidates candidates_v3.no_agentdog.jsonl \
        --gold gold_cases.jsonl \
        --out-dir dataset_v1
"""
import argparse
import collections
import hashlib
import json
import random
from pathlib import Path

FACETS = [
    "data_exposure",
    "destructive",
    "financial",
    "credential_identity",
    "authorization_scope",
    "injection_hierarchy",
    "code_execution",
    "deception_social",
    "physical_cyber_harm",
    "policy_legal",
]

SPLIT_SALT = "split-v1"
SEED = 20261111

# Quotas from DATA_SPEC.md section 6, section 7 and section 15.
TRAIN_FRACS = (0.70, 0.15, 0.15)
TRAIN_UNSAFE_FRAC = 0.27
TRAIN_MIN = 12000
TRAIN_MAX = 15000
EVAL_NATURAL = 2000
EVAL_NATURAL_UNSAFE_FRAC = 0.17
EVAL_BALANCED_UNSAFE = 250
EVAL_BALANCED_HARD = 250


def sha256_text(value):
    return hashlib.sha256(value.encode()).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_annotations(path):
    result = {}
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("status") != "ok":
                continue
            result[str(row["key"])] = row
    return result


def load_gold_keys(path):
    keys = set()
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                keys.add(str(json.loads(line)["key"]))
    return keys


def proportional(counts, total):
    """Split `total` across strata proportionally to `counts` (largest remainder)."""
    population = sum(counts.values())
    if population == 0 or total <= 0:
        return {name: 0 for name in counts}
    exact = {name: total * size / population for name, size in counts.items()}
    result = {name: int(value) for name, value in exact.items()}
    leftover = total - sum(result.values())
    order = sorted(exact, key=lambda name: exact[name] - int(exact[name]), reverse=True)
    for name in order[:leftover]:
        result[name] += 1
    return result


def take(keys, limit, rng):
    if limit <= 0:
        return []
    if limit >= len(keys):
        return sorted(keys)
    return sorted(rng.sample(sorted(keys), limit))


def take_by_source(keys, limit, rng):
    """Sample `limit` rows keeping the source mix of `keys`."""
    if limit <= 0:
        return []
    if limit >= len(keys):
        return sorted(keys)
    by_source = collections.defaultdict(list)
    for key in keys:
        by_source[META[key]["source"]].append(key)
    quota = proportional({name: len(items) for name, items in by_source.items()}, limit)
    picked = []
    for name, size in sorted(quota.items()):
        picked.extend(take(by_source[name], size, rng))
    return sorted(picked)


def role_of(verdict, group_has_unsafe):
    if verdict == "unsafe":
        return "unsafe"
    if verdict == "uncertain":
        return "uncertain"
    return "hard_negative" if group_has_unsafe else "ordinary_safe"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--ood-pool", type=Path, help="full candidate pool, resolves OOD rows absent from --candidates")
    parser.add_argument("--prompt-file", type=Path, default=Path(__file__).with_name("flash_next") / "prompts" / "prompt.txt")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    annotations = load_annotations(args.annotations)
    gold_keys = load_gold_keys(args.gold)

    global META
    META = {}
    dropped = collections.Counter()
    for key, row in annotations.items():
        if key in gold_keys:
            dropped["gold"] += 1
            continue
        sources = row.get("sources") or []
        if "AgentDoG" in sources:
            dropped["agentdog_kept_for_ood"] += 1
        parsed = row.get("parsed") or {}
        META[key] = {
            "group": str(row.get("group")),
            "source": sources[0] if sources else "unknown",
            "verdict": parsed.get("verdict"),
            "facets": list(parsed.get("facets") or []),
            "ood": "AgentDoG" in sources,
        }
    print(json.dumps({"annotations_ok": len(annotations), "dropped": dropped}))

    pool = [key for key, meta in META.items() if not meta["ood"]]
    ood = [key for key, meta in META.items() if meta["ood"]]

    group_has_unsafe = collections.defaultdict(bool)
    for key in META:
        if META[key]["verdict"] == "unsafe":
            group_has_unsafe[META[key]["group"]] = True

    # Group-disjoint split, stratified by whether the group holds an unsafe action
    # so every split gets the same unsafe prevalence.
    strata = {True: [], False: []}
    for group in sorted({META[key]["group"] for key in pool}):
        strata[group_has_unsafe[group]].append(group)
    split_of_group = {}
    for groups in strata.values():
        groups.sort(key=lambda group: sha256_text(SPLIT_SALT + ":" + group))
        train_end = int(len(groups) * TRAIN_FRACS[0])
        dev_end = int(len(groups) * (TRAIN_FRACS[0] + TRAIN_FRACS[1]))
        for group in groups[:train_end]:
            split_of_group[group] = "train"
        for group in groups[train_end:dev_end]:
            split_of_group[group] = "dev"
        for group in groups[dev_end:]:
            split_of_group[group] = "test"

    def bucket(split):
        return [key for key in pool if split_of_group[META[key]["group"]] == split]

    rng = random.Random(args.seed)
    selected = collections.defaultdict(list)  # split -> [(key, slice)]

    # --- train: all unsafe in the bucket, 1:1 hard negatives, ordinary safe to 27%.
    train = bucket("train")
    train_unsafe = [key for key in train if META[key]["verdict"] == "unsafe"]
    train_hard = [key for key in train if role_of(META[key]["verdict"], group_has_unsafe[META[key]["group"]]) == "hard_negative"]
    train_ordinary = [key for key in train if role_of(META[key]["verdict"], group_has_unsafe[META[key]["group"]]) == "ordinary_safe"]

    train_total = max(TRAIN_MIN, round(len(train_unsafe) / TRAIN_UNSAFE_FRAC))
    n_hard = min(len(train_hard), len(train_unsafe))
    n_ordinary = max(0, train_total - len(train_unsafe) - n_hard)
    for key in train_unsafe:
        selected["train"].append((key, "train"))
    for key in take_by_source(train_hard, n_hard, rng):
        selected["train"].append((key, "train"))
    for key in take_by_source(train_ordinary, n_ordinary, rng):
        selected["train"].append((key, "train"))

    # --- dev/test: natural-prevalence slice plus a balanced-by-facet slice.
    for split in ("dev", "test"):
        rows = bucket(split)
        by_verdict = collections.defaultdict(list)
        for key in rows:
            by_verdict[META[key]["verdict"]].append(key)

        n_unsafe = round(EVAL_NATURAL * EVAL_NATURAL_UNSAFE_FRAC)
        natural_unsafe = take_by_source(by_verdict["unsafe"], n_unsafe, rng)
        natural_uncertain = take_by_source(by_verdict["uncertain"], round(EVAL_NATURAL * len(by_verdict["uncertain"]) / max(len(rows), 1)), rng)
        natural_safe = take_by_source(
            by_verdict["safe"], EVAL_NATURAL - len(natural_unsafe) - len(natural_uncertain), rng
        )
        used = set(natural_unsafe) | set(natural_uncertain) | set(natural_safe)
        for key in natural_unsafe + natural_uncertain + natural_safe:
            selected[split].append((key, "natural"))

        # Balanced slice: unsafe round-robin over facets, each paired with a safe
        # sibling from the same group when one exists.
        by_facet = {facet: [] for facet in FACETS}
        for key in by_verdict["unsafe"]:
            if key in used:
                continue
            for facet in META[key]["facets"] or ["none"]:
                by_facet.setdefault(facet, []).append(key)
        for keys in by_facet.values():
            keys.sort()

        balanced_unsafe = []
        cursors = {facet: 0 for facet in by_facet}
        while len(balanced_unsafe) < EVAL_BALANCED_UNSAFE:
            progressed = False
            for facet in sorted(by_facet):
                keys = by_facet[facet]
                while cursors[facet] < len(keys) and keys[cursors[facet]] in used:
                    cursors[facet] += 1
                if cursors[facet] < len(keys) and len(balanced_unsafe) < EVAL_BALANCED_UNSAFE:
                    key = keys[cursors[facet]]
                    cursors[facet] += 1
                    used.add(key)
                    balanced_unsafe.append(key)
                    progressed = True
            if not progressed:
                break

        siblings = collections.defaultdict(list)
        for key in rows:
            if META[key]["verdict"] == "safe" and key not in used:
                siblings[META[key]["group"]].append(key)
        balanced_hard = []
        for key in balanced_unsafe:
            for candidate in sorted(siblings.get(META[key]["group"], [])):
                if candidate not in used:
                    used.add(candidate)
                    balanced_hard.append(candidate)
                    break
        if len(balanced_hard) < EVAL_BALANCED_HARD:
            spare = [key for key in rows if META[key]["verdict"] == "safe" and key not in used]
            for key in take(spare, EVAL_BALANCED_HARD - len(balanced_hard), rng):
                used.add(key)
                balanced_hard.append(key)

        for key in balanced_unsafe + balanced_hard:
            selected[split].append((key, "balanced"))

    for key in ood:
        selected["ood"].append((key, "ood"))

    wanted = {key for entries in selected.values() for key, _ in entries}
    print(json.dumps({"selected": {split: len(entries) for split, entries in selected.items()}, "wanted": len(wanted)}))

    # --- pass 2: stream the pool, emit selected rows with their visible state.
    slice_of = {key: name for entries in selected.values() for key, name in entries}
    split_of_key = {key: split for split, entries in selected.items() for key, _ in entries}
    written = collections.Counter()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    sinks = {split: (args.out_dir / f"{split}.jsonl").open("w", encoding="utf-8") for split in selected}

    try:
        pending = set(wanted)
        for pool_path in (args.candidates, args.ood_pool):
            if pool_path is None or not pending:
                continue
            with pool_path.open("r", encoding="utf-8") as stream:
                for line in stream:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    key = str(row["key"])
                    if key not in pending:
                        continue
                    state = row.get("state")
                    if isinstance(state, str):
                        state = json.loads(state)
                    annotation = annotations[key]
                    parsed = annotation.get("parsed") or {}
                    verdict = META[key]["verdict"]
                    split = split_of_key[key]
                    record = {
                        "key": key,
                        "group": META[key]["group"],
                        "split": split,
                        "slice": slice_of[key],
                        "input": state,
                        "label": None if verdict == "uncertain" else int(verdict == "unsafe"),
                        "uncertain": verdict == "uncertain",
                        "facets": META[key]["facets"],
                        "severity": parsed.get("severity"),
                        "reversibility": parsed.get("reversibility"),
                        "actor": parsed.get("actor"),
                        "authorization": parsed.get("authorization"),
                        "evidence": parsed.get("evidence_sufficiency"),
                        "role": role_of(verdict, group_has_unsafe[META[key]["group"]]),
                        "sources": annotation.get("sources") or [],
                        "input_sha256": annotation.get("input_sha256_prompt") or row.get("input_sha256"),
                        "annotation": {
                            "model": annotation.get("model"),
                            "prompt": "prompt.txt",
                            "prompt_sha256": annotation.get("prompt_sha256"),
                            "confidence": parsed.get("confidence"),
                            "reason": parsed.get("reason"),
                            "status": "silver",
                        },
                    }
                    sinks[split].write(json.dumps(record, ensure_ascii=False) + "\n")
                    written[split] += 1
                    pending.discard(key)
    finally:
        for sink in sinks.values():
            sink.close()
    if pending:
        raise SystemExit(f"unresolved keys: {len(pending)}")

    print(json.dumps({"written": dict(written)}))

    # --- manifest and stats.
    def summarize(split):
        path = args.out_dir / f"{split}.jsonl"
        rows = [json.loads(line) for line in path.open("r", encoding="utf-8")]
        facets = collections.Counter()
        for row in rows:
            for facet in row["facets"]:
                facets[facet] += 1
        labels = collections.Counter("null" if row["label"] is None else str(row["label"]) for row in rows)
        return {
            "rows": len(rows),
            "labels": dict(labels),
            "roles": dict(collections.Counter(row["role"] for row in rows)),
            "slices": dict(collections.Counter(row["slice"] for row in rows)),
            "sources": dict(collections.Counter(source for row in rows for source in row["sources"])),
            "facets": dict(sorted(facets.items(), key=lambda item: -item[1])),
            "groups": len({row["group"] for row in rows}),
            "sha256": sha256_file(path),
        }

    manifest = {
        "version": "1.0",
        "created": "2026-11-11",
        "spec": "docs/DATA_SPEC.md",
        "judge": {
            "model": "qwen3.8-flash-next",
            "prompt": "prompts/prompt.txt",
            "prompt_sha256": sha256_file(args.prompt_file),
            "status": "silver",
        },
        "inputs": {
            "annotations": {"path": str(args.annotations), "sha256": sha256_file(args.annotations)},
            "candidates": {"path": str(args.candidates), "sha256": sha256_file(args.candidates)},
            "gold": {"path": str(args.gold), "sha256": sha256_file(args.gold), "keys": len(gold_keys)},
        },
        "build_script": {"path": str(Path(__file__)), "sha256": sha256_file(Path(__file__))},
        "filters": {
            "gold_keys_excluded": len(gold_keys),
            "agentdog_excluded_from_train": dropped["agentdog_kept_for_ood"],
            "agentdog_policy": "OOD only, never train",
        },
        "split_protocol": {
            "unit": "group",
            "salt": SPLIT_SALT,
            "seed": args.seed,
            "fractions": {"train": TRAIN_FRACS[0], "dev": TRAIN_FRACS[1], "test": TRAIN_FRACS[2]},
            "stratified_by": "group_contains_unsafe",
        },
        "quotas": {
            "train_unsafe_frac": TRAIN_UNSAFE_FRAC,
            "eval_natural_rows": EVAL_NATURAL,
            "eval_natural_unsafe_frac": EVAL_NATURAL_UNSAFE_FRAC,
            "eval_balanced_unsafe": EVAL_BALANCED_UNSAFE,
            "eval_balanced_hard_negatives": EVAL_BALANCED_HARD,
        },
        "splits": {split: summarize(split) for split in ("train", "dev", "test", "ood")},
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.out_dir / "facets.json").write_text(
        json.dumps({"version": "v1", "facets": FACETS}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({split: manifest["splits"][split]["rows"] for split in manifest["splits"]}))


if __name__ == "__main__":
    main()
