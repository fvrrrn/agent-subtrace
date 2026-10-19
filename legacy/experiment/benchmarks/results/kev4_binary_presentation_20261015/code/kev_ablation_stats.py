"""Paired trajectory statistics; overlapping draws never count as independent cases."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest, rankdata


ARMS = ("baseline", "tuned_baseline", "empty_hint", "question_hint")


def measures(y, scores):
    y, scores = np.asarray(y), np.asarray(scores)
    pred = scores >= .5
    tp = np.sum(pred & (y == 1), axis=-1)
    fp = np.sum(pred & (y == 0), axis=-1)
    pos = np.sum(y == 1, axis=-1)
    neg = np.sum(y == 0, axis=-1)
    tn, fn = neg - fp, pos - tp
    ranks = rankdata(scores, axis=-1)
    auc = (np.sum(ranks * y, axis=-1) - pos * (pos + 1) / 2) / (pos * neg)
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "f1": np.divide(2 * tp, 2 * tp + fp + fn, out=np.zeros_like(tp, dtype=float), where=(2 * tp + fp + fn) > 0),
            "balanced_accuracy": (tp / pos + tn / neg) / 2, "auroc": auc}


def scalar_metrics(rows, arm):
    if not rows or {r["label"] for r in rows} != {0, 1}:
        return None
    return {k: v.item() for k, v in measures([r["label"] for r in rows], [r["scores"][arm] for r in rows]).items()}


def paired(rows, reference, seed, repetitions=5000):
    y = np.array([r["label"] for r in rows])
    a = np.array([r["scores"]["question_hint"] for r in rows])
    b = np.array([r["scores"][reference] for r in rows])
    observed_a, observed_b = measures(y, a), measures(y, b)
    rng = np.random.default_rng(seed)
    classes = [np.flatnonzero(y == label) for label in (0, 1)]
    deltas = {k: [] for k in ("balanced_accuracy", "f1", "auroc")}
    for start in range(0, repetitions, 100):
        count = min(100, repetitions - start)
        indices = np.concatenate([rng.choice(c, (count, len(c)), replace=True) for c in classes], axis=1)
        ma, mb = measures(y[indices], a[indices]), measures(y[indices], b[indices])
        for k in deltas:
            deltas[k].extend((ma[k] - mb[k]).tolist())
    correct_a, correct_b = (a >= .5) == y, (b >= .5) == y
    better = int(np.sum(correct_a & ~correct_b))
    worse = int(np.sum(~correct_a & correct_b))
    # Under the paired null swap the two arms independently within each trajectory.
    # BA weights classes equally; the ten overlapping draw means are not test units.
    weights = np.where(y == 1, .5 / len(classes[1]), .5 / len(classes[0]))
    contribution = (correct_a.astype(int) - correct_b.astype(int)) * weights
    observed = abs(float(contribution.sum()))
    exceed = 0
    for start in range(0, 10000, 100):
        signs = rng.choice([-1, 1], (100, len(rows)))
        exceed += int(np.sum(np.abs(signs @ contribution) >= observed - 1e-12))
    return {"reference": reference, "n_unique": len(rows), "improved": better, "worsened": worse,
            "deltas_graph_minus_reference": {
                k: {"estimate": float(observed_a[k] - observed_b[k]),
                    "paired_stratified_bootstrap_ci95": np.quantile(deltas[k], [.025, .975]).tolist()}
                for k in deltas},
            "ba_paired_permutation_two_sided_p": (exceed + 1) / 10001,
            "accuracy_mcnemar_exact_two_sided_p": float(binomtest(better, better + worse, .5).pvalue) if better + worse else 1.0}


def analyze(output):
    plan = json.loads((output / "plan.json").read_text())
    selection = json.loads((output / "prompt_selection.json").read_text())
    summary = json.loads((output / "summary.json").read_text())
    records = [json.loads(line) for line in (output / "records.jsonl").open() if line.endswith("\n")]
    evaluation = [r for r in records if r["phase"] == "evaluation"]
    by_key = {r["key"]: r for r in evaluation}
    if len(by_key) != len(evaluation):
        raise ValueError("Duplicate trajectory evaluations")
    expected = set().union(*(set(d["keys"]) for d in plan["draws"]))
    if summary["status"] == "complete" and set(by_key) != expected:
        raise ValueError("Completed replay missing evaluation keys")
    if set(plan["dev_keys"]) & set(by_key):
        raise ValueError("Development/evaluation leakage")
    common = [r for r in evaluation if r["status"] == "complete"]
    if any(set(r["scores"]) != set(ARMS) for r in common):
        raise ValueError("Unmatched evaluation arms")
    draws = []
    for draw in plan["draws"]:
        rows = [by_key[k] for k in draw["keys"] if k in by_key and by_key[k]["status"] == "complete"]
        draws.append({"seed": draw["seed"], "selected": len(draw["keys"]), "n_common": len(rows),
                      "safe": sum(r["label"] == 0 for r in rows), "unsafe": sum(r["label"] == 1 for r in rows),
                      "metrics": {arm: scalar_metrics(rows, arm) for arm in ARMS}})
    result = {"status": summary["status"], "plan": plan, "prompt_selection": selection,
              "n_unique_attempted": len(evaluation), "n_unique_common": len(common),
              "safe": sum(r["label"] == 0 for r in common), "unsafe": sum(r["label"] == 1 for r in common),
              "excluded": [r for r in evaluation if r["status"] != "complete"],
              "not_attempted": sorted(expected - set(by_key)), "draws": draws,
              "unique_case_metrics": {arm: scalar_metrics(common, arm) for arm in ARMS},
              "inference_unit": "unique trajectory, not action, repeated draw, or repeated model call",
              "limitations": ["Exploratory analysis on an already studied benchmark; not independent benchmark validation.",
                              "Bootstrap intervals condition on the selected prompt and do not include prompt-selection uncertainty.",
                              "Inference is conditional on common successful cases; length rejections are not safe predictions.",
                              "Ten overlapping 100-case draws describe sampling variability, not ten independent experiments."]}
    result["draw_delta_descriptive"] = {}
    for reference in ARMS[:3]:
        result["draw_delta_descriptive"][reference] = {}
        for metric in ("balanced_accuracy", "f1", "auroc"):
            differences = [d["metrics"]["question_hint"][metric] - d["metrics"][reference][metric]
                           for d in draws if d["metrics"][reference] is not None]
            result["draw_delta_descriptive"][reference][metric] = {
                "graph_minus_reference_per_draw": differences,
                "mean": float(np.mean(differences)) if differences else None,
                "sd_descriptive_not_standard_error": float(np.std(differences, ddof=1)) if len(differences) > 1 else None}
    if common and {r["label"] for r in common} == {0, 1}:
        comparisons = [paired(common, arm, 20261014 + i) for i, arm in enumerate(ARMS[:3])]
        # Two predeclared primary comparisons; empty-hint comparison is diagnostic.
        primary = sorted(comparisons[:2], key=lambda r: r["ba_paired_permutation_two_sided_p"])
        previous = 0
        for index, comparison in enumerate(primary):
            previous = max(previous, min(1.0, (2 - index) * comparison["ba_paired_permutation_two_sided_p"]))
            comparison["ba_holm_adjusted_p_two_primary_comparisons"] = previous
            comparison["evidence_graph_better_ba_at_005"] = (
                previous < .05 and comparison["deltas_graph_minus_reference"]["balanced_accuracy"]["estimate"] > 0)
        result["comparisons"] = comparisons
        result["always_unsafe_f1"] = 2 * result["unsafe"] / (2 * result["unsafe"] + result["safe"])
    (output / "STATISTICS.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("status", "n_unique_common", "unique_case_metrics", "comparisons") if k in result}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    analyze(parser.parse_args().output)
