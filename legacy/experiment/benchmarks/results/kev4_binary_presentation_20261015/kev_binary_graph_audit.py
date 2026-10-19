"""Compare saved binary LINK predictions with frozen exploratory blind annotations."""

import argparse
import hashlib
import itertools
import json
from pathlib import Path


def metrics(rows):
    tp = sum(r["label"] == 1 and r["prediction"] for r in rows)
    fp = sum(r["label"] == 0 and r["prediction"] for r in rows)
    fn = sum(r["label"] == 1 and not r["prediction"] for r in rows)
    tn = len(rows) - tp - fp - fn
    positive = [r["score"] for r in rows if r["label"] == 1]
    negative = [r["score"] for r in rows if r["label"] == 0]
    return {"n": len(rows), "positive_reference_pairs": len(positive), "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0,
            "auroc": sum((a > b) + .5 * (a == b) for a in positive for b in negative) / (len(positive) * len(negative)) if positive and negative else None}


def main(root):
    packet = json.loads((root / "AUDIT_BLIND.json").read_text())
    annotations = json.loads((root / "AUDIT_ANNOTATIONS.json").read_text())
    threshold = json.loads((root / "THRESHOLD.json").read_text())["threshold"]
    labels = {r["pair_id"]: r for r in annotations["annotations"]}
    if len(labels) != len(packet["sample"]) or set(labels) != {p["pair_id"] for p in packet["sample"]}:
        raise ValueError("Blind packet/annotation coverage mismatch")
    by_key = {r["key"]: r for i in range(2) for line in (root / f"link_{i}/records.jsonl").open() for r in [json.loads(line)]}
    matched, unscored = [], []
    for pair in packet["sample"]:
        context = packet["contexts"][pair["context_id"]]
        record = by_key[context["key"]]
        entry = {**pair, "key": context["key"], **labels[pair["pair_id"]]}
        if record["status"] != "complete":
            unscored.append({**entry, "reason": record.get("error", record["status"])})
            continue
        step = next(s for s in record["steps"] if s["id"] == pair["target"])
        score = step["scores"][pair["source"]]
        matched.append({**entry, "score": score, "prediction": score >= threshold})
    certain = [r for r in matched if r["label"] in (0, 1)]
    uncertain = [r for r in matched if r["label"] == "uncertain"]
    sensitivity = [metrics(certain + [{**r, "label": label} for r, label in zip(uncertain, assignment)])
                   for assignment in itertools.product((0, 1), repeat=len(uncertain))]
    result = {"reference_type": annotations["reference_type"], "reviewer": annotations["reviewer"],
              "threshold": threshold, "n_sampled": len(packet["sample"]), "n_scored": len(matched),
              "n_certain_scored": len(certain), "n_uncertain_scored": len(uncertain),
              "exploratory_metrics_certain_only": metrics(certain), "unscored": unscored, "pairs": matched,
              "uncertain_assignment_sensitivity_not_confidence_intervals": {
                  key: [min(values), max(values)] if values else None
                  for key in ("precision", "recall", "f1")
                  for values in [[m[key] for m in sensitivity if m[key] is not None]]},
              "frozen_input_sha256": {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                                      for name in ("AUDIT_BLIND.json", "AUDIT_ANNOTATIONS.json", "THRESHOLD.json")},
              "limitations": annotations["limitations"] + ["Not official ATBench edge metrics or independent expert assessment.",
                                                           "Small, clustered sample; do not interpret a high score as established graph accuracy."]}
    (root / "GRAPH_AUDIT.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("n_sampled", "n_scored", "n_certain_scored", "n_uncertain_scored", "exploratory_metrics_certain_only")}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    main(parser.parse_args().output)
