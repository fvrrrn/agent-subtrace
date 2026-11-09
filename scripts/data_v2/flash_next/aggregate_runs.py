"""Aggregate prompt-evaluation runs and apply the RRSI selection rule.

Reads eval_<tag>_<split>.jsonl files produced by eval_prompt.py and prints
per-config metrics plus the selection decision. The rule is fixed before
looking at the numbers:

    delta      = max - min of S over repeated runs of the unchanged baseline
    eligible   = S(H') >= S(v1) - delta  and  dC <= beta0 + beta1 * max(dS, 0)
    confirmed  = S(H') - S(v1) > delta
    S*         = best eligible candidate, ties broken by lower cost

S is 3-class macro F1 over {safe, unsafe, uncertain}; dS is absolute,
dC is relative token cost, beta0 = 0.10, beta1 = 1.0.

Usage:
    python aggregate_runs.py <dir> [tag ...]
"""
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

BETA0 = 0.10
BETA1 = 1.0
BASELINE_PREFIX = "v1"


def macro_f1(gold, pred, n_classes=3):
    scores = []
    for cls in range(n_classes):
        tp = sum(1 for g, p in zip(gold, pred) if g == cls and p == cls)
        fp = sum(1 for g, p in zip(gold, pred) if g != cls and p == cls)
        fn = sum(1 for g, p in zip(gold, pred) if g == cls and p != cls)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        scores.append(f1)
    return sum(scores) / len(scores)


def metrics(rows):
    ok = [r for r in rows if r["status"] == "ok"]
    gold = [r["gold"] for r in ok]
    pred = [r["pred"] for r in ok]
    if not gold:
        return {"n": 0}
    covered = [(g, p) for g, p in zip(gold, pred) if p != 2]
    binary = [(g, p) for g, p in zip(gold, pred) if g != 2]
    unsafe = [(g, p) for g, p in zip(gold, pred) if g == 1]
    safe = [(g, p) for g, p in zip(gold, pred) if g == 0]
    predicted_unsafe = [(g, p) for g, p in zip(gold, pred) if p == 1]
    cost = [r["tokens"] for r in ok if r.get("tokens") is not None]
    return {
        "n": len(gold),
        "status": dict(Counter(r["status"] for r in rows)),
        "accuracy3": round(sum(g == p for g, p in zip(gold, pred)) / len(gold), 4),
        "macro_f1_3": round(macro_f1(gold, pred), 4),
        "binary_accuracy_all": round(sum(g == p for g, p in binary) / len(binary), 4) if binary else None,
        "binary_accuracy_covered": round(sum(g == p for g, p in covered) / len(covered), 4) if covered else None,
        "coverage": round(len(covered) / len(gold), 4),
        "unsafe_recall": round(sum(p == 1 for g, p in unsafe) / len(unsafe), 4) if unsafe else None,
        "unsafe_precision": round(sum(g == 1 for g, p in predicted_unsafe) / len(predicted_unsafe), 4) if predicted_unsafe else None,
        "safe_recall": round(sum(p == 0 for g, p in safe) / len(safe), 4) if safe else None,
        "false_positive_rate": round(sum(p == 1 for g, p in safe) / len(safe), 4) if safe else None,
        "false_negative_rate": round(sum(p == 0 for g, p in unsafe) / len(unsafe), 4) if unsafe else None,
        "cost_mean": round(statistics.mean(cost), 2) if cost else None,
    }


def discover(directory, split):
    return sorted(p.stem[len("eval_"):-len(f"_{split}")] for p in directory.glob(f"eval_*_{split}.jsonl"))


def main():
    directory = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    split = "dev"
    tags = sys.argv[2:] or discover(directory, split)

    results = {}
    for tag in tags:
        path = directory / f"eval_{tag}_{split}.jsonl"
        if not path.exists():
            print(f"missing {path}")
            continue
        results[tag] = (metrics([json.loads(line) for line in path.open() if line.strip()]),)

    print(f"{'tag':10s} {'S=f1_3':>7s} {'acc3':>6s} {'bin_all':>7s} {'u_rec':>6s} "
          f"{'u_prec':>6s} {'fpr':>6s} {'fnr':>6s} {'cov':>6s} {'tokens':>8s}")
    for tag, (m,) in results.items():
        print(f"{tag:10s} {m['macro_f1_3']:7.4f} {m['accuracy3']:6.4f} "
              f"{str(m['binary_accuracy_all']):>7s} {str(m['unsafe_recall']):>6s} "
              f"{str(m['unsafe_precision']):>6s} {str(m['false_positive_rate']):>6s} "
              f"{str(m['false_negative_rate']):>6s} {m['coverage']:6.3f} {m['cost_mean']:8.0f}")

    base_tags = [t for t in tags if t.startswith(BASELINE_PREFIX) and t in results]
    if not base_tags:
        return
    base_scores = [results[t][0]["macro_f1_3"] for t in base_tags]
    base_costs = [results[t][0]["cost_mean"] for t in base_tags]
    delta = max(base_scores) - min(base_scores)
    base_s = statistics.mean(base_scores)
    base_c = statistics.mean(base_costs)
    print(f"\nbaseline (n={len(base_scores)}): S={base_s:.4f} cost={base_c:.0f} "
          f"runs={[round(s, 4) for s in base_scores]}")
    print(f"noise band delta = {delta:.4f}")

    groups = {}
    for tag in tags:
        if tag.startswith(BASELINE_PREFIX) and tag in results:
            continue
        if tag in results:
            groups.setdefault(tag.split("_")[0], []).append(tag)

    print("\nper-config spread:")
    worst = delta
    for name, group in sorted(groups.items()):
        ss = [results[t][0]["macro_f1_3"] for t in group]
        spread = max(ss) - min(ss)
        worst = max(worst, spread)
        print(f"{name:6s} n={len(ss)} S={statistics.mean(ss):.4f} "
              f"runs={[round(s, 4) for s in ss]} spread={spread:.4f}")
    print(f"worst observed spread = {worst:.4f}")

    print("\nselection:")
    for name, group in sorted(groups.items()):
        s = statistics.mean(results[t][0]["macro_f1_3"] for t in group)
        c = statistics.mean(results[t][0]["cost_mean"] for t in group)
        ds = s - base_s
        dc = (c - base_c) / base_c if base_c else 0.0
        quality_ok = s >= base_s - delta
        cost_ok = dc <= BETA0 + BETA1 * max(ds, 0.0)
        verdict = "ACCEPT" if (quality_ok and cost_ok) else "REJECT"
        flag = "CONFIRMED" if ds > delta else ("within-noise" if ds >= 0 else "regression")
        print(f"{name:6s} S={s:.4f} dS={ds:+.4f} cost={c:.0f} dC={dc:+.3f} "
              f"| quality_ok={quality_ok} cost_ok={cost_ok} -> {verdict} ({flag}) "
              f"above_worst_spread={ds > worst}")


if __name__ == "__main__":
    main()
