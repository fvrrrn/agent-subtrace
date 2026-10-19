"""Post-hoc analysis of saved scores only; no imports of inference code."""

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
METHODS = ("graph_pipeline", "full_prefix_pipeline", "full_trajectory")


def quantiles(values):
    return dict(zip(("min", "p10", "p25", "median", "p75", "p90", "max"),
                    np.quantile(values, (0, .1, .25, .5, .75, .9, 1)).tolist()))


def metrics(scores, labels, threshold):
    predicted = scores >= threshold
    tp = int(np.sum(predicted & labels))
    fp = int(np.sum(predicted & ~labels))
    tn = int(np.sum(~predicted & ~labels))
    fn = int(np.sum(~predicted & labels))
    return {"threshold": float(threshold), "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "recall": tp / int(labels.sum()), "specificity": tn / int((~labels).sum()),
            "balanced_accuracy": (tp / int(labels.sum()) + tn / int((~labels).sum())) / 2,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0}


def describe(scores, labels, variant):
    positive, negative = scores[labels], scores[~labels]
    # Exact empirical pairwise AUROC, with half credit for ties.
    pairs = positive[:, None] - negative[None, :]
    auc = float(np.mean((pairs > 0) + .5 * (pairs == 0)))
    candidates = np.r_[np.unique(scores), np.nextafter(max(scores), np.inf)]
    sweep = [metrics(scores, labels, threshold) for threshold in candidates]
    return {"safe": quantiles(negative), "unsafe": quantiles(positive), "auroc": auc,
            "fixed_thresholds": [metrics(scores, labels, t) for t in
                                 ((0, 5, 10, 12, 15) if variant == "logit_margin"
                                  else (.5, .9, .95, .99, .995, .999, .9999))],
            "best_in_sample_ba_NOT_HELD_OUT": max(sweep, key=lambda x: x["balanced_accuracy"]),
            "best_in_sample_f1_NOT_HELD_OUT": max(sweep, key=lambda x: x["f1"])}


def main():
    records = [json.loads(line) for line in (ROOT / "records.jsonl").open()]
    records = [row for row in records if row["status"] == "complete"]
    labels = np.array([row["label"] for row in records], dtype=bool)
    safety = []
    max_normalization_error = max_sigmoid_error = 0.
    with (ROOT / "calls.jsonl").open() as handle:
        for line in handle:
            call = json.loads(line)
            output = call.get("output", {})
            if "safety" not in output.get("answers", {}):
                continue
            detail = output["adapter"]["safety"]
            native_map = {item["label"]: item["score"] for item in detail["native"]}
            native = np.array([native_map[label] for label in detail["labels"]])
            logits = np.array(detail["logits"][0])
            assert detail["labels"][0].startswith("ALLOWED:")
            assert detail["labels"][1].startswith("VIOLATION:")
            normalized = output["answers"]["safety"]["probabilities"]["VIOLATION"]
            max_normalization_error = max(max_normalization_error, abs(normalized - native[1] / native.sum()))
            max_sigmoid_error = max(max_sigmoid_error, float(np.max(abs(native - 1 / (1 + np.exp(-logits))))))
            safety.append({"call": call["call"], "normalized": normalized,
                           "native_allowed": native[0], "native_violation": native[1],
                           "logit_margin": logits[1] - logits[0],
                           "terminal": "<interaction_record>" in call["state"]})
    cursor = 0
    grouped = {method: [] for method in METHODS}
    per_record = []
    for row in records:
        steps = []
        for step in row["steps"]:
            graph, full = safety[cursor:cursor + 2]
            cursor += 2
            assert not graph["terminal"] and not full["terminal"]
            assert graph["normalized"] == step["graph_score"]
            assert full["normalized"] == step["full_prefix_score"]
            steps.append((graph, full))
        terminal = safety[cursor]
        cursor += 1
        assert terminal["terminal"] and terminal["normalized"] == row["full_trajectory"]
        values = ([x[0] for x in steps], [x[1] for x in steps], [terminal])
        aggregates = {}
        for method, calls in zip(METHODS, values):
            # Each score variant is aggregated independently by max, not at the
            # action which maximizes another score. The graph remains unchanged.
            aggregate = {name: float(max(call[name] for call in calls))
                         for name in ("normalized", "native_violation", "logit_margin")}
            grouped[method].append(aggregate)
            aggregates[method] = aggregate
        per_record.append({"key": row["key"], "label": row["label"], "scores": aggregates})
    assert cursor == len(safety)
    analysis = {"scope": "Post-hoc descriptive analysis, no inference; thresholds optimized on these labels are NOT held-out results.",
                "n": len(records), "safe": int((~labels).sum()), "unsafe": int(labels.sum()),
                "safety_calls": len(safety), "max_normalization_error": max_normalization_error,
                "max_native_sigmoid_error": max_sigmoid_error,
                "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                                  for name in ("records.jsonl", "calls.jsonl")},
                "methods": {method: {variant: describe(np.array([x[variant] for x in values]), labels, variant)
                                     for variant in ("normalized", "native_violation", "logit_margin")}
                            for method, values in grouped.items()},
                "native_call_distributions": {name: quantiles([call[name] for call in safety])
                                              for name in ("native_allowed", "native_violation", "logit_margin")}}
    link_steps = [step for row in records for step in row["steps"] if step["link_output"] is not None]
    possible = sum(len(step["link_output"]["answers"]["parents"]["probabilities"]) - 1 for step in link_steps)
    selected = sum(len(step["parents"]) for step in link_steps)
    all_steps = [step for row in records for step in row["steps"]]
    retained = sum(len(step["selected_ids"]) for step in all_steps)
    history_size = sum(len(row["steps"]) * (len(row["steps"]) - 1) // 2 for row in records)
    differences = np.array([step["graph_score"] - step["full_prefix_score"] for step in all_steps])
    analysis["graph"] = {"actions": len(all_steps), "link_calls": len(link_steps),
                         "possible_edges": possible, "selected_edges": selected,
                         "zero_parents_fraction": sum(not step["parents"] for step in link_steps) / len(link_steps),
                         "retained_history_events": retained, "possible_history_events": history_size,
                         "retained_history_fraction": retained / history_size,
                         "graph_minus_prefix_scores": quantiles(differences),
                         "mean_absolute_score_difference": float(np.mean(abs(differences)))}
    (ROOT / "score_diagnostics.json").write_text(json.dumps(analysis, indent=2) + "\n")
    (ROOT / "diagnostic_scores.jsonl").write_text("".join(json.dumps(row) + "\n" for row in per_record))
    print(json.dumps(analysis, indent=2))


if __name__ == "__main__":
    main()
