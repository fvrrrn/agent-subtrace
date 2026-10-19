"""Threshold-free graph evaluation and matched choice/NOUL inference. JSON only."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random

import numpy as np


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def arrays(points):
    y = np.array([p["gold"] for p in points], dtype=int)
    s = np.array([p["score"] for p in points], dtype=float)
    if not len(y) or not np.isfinite(s).all() or not np.isin(y, [0, 1]).all():
        raise ValueError("Expected finite scores and binary gold for nonempty edge set")
    return y, s


def curve_arrays(y, s):
    order = np.argsort(-s, kind="stable")
    ends = np.r_[np.flatnonzero(np.diff(s[order])), len(y) - 1]
    tp = np.cumsum(y[order])[ends]
    selected = ends + 1
    positives = int(y.sum())
    recall = tp / positives if positives else np.zeros(len(tp))
    precision = tp / selected
    return s[order][ends], tp, selected, precision, recall


def ranking(y, s):
    _, tp, selected, precision, recall = curve_arrays(y, s)
    positives, n = int(y.sum()), len(y)
    ap = float(np.sum(np.diff(np.r_[0., recall]) * precision)) if positives else None
    fpr = (selected - tp) / (n - positives) if n > positives else np.zeros(len(tp))
    auc = float(np.sum(np.diff(np.r_[0., fpr]) * (np.r_[0., recall][:-1] + recall) / 2)) if 0 < positives < n else None
    return {"average_precision": ap, "roc_auc": auc, "positive_prevalence": positives / n,
            "ap_over_prevalence": ap / (positives / n) if positives else None,
            "pr_auc_trapezoid_diagnostic": float(np.sum(np.diff(np.r_[0., recall]) * (np.r_[1., precision][:-1] + precision) / 2)) if positives else None,
            "n_pairs": n, "n_gold_edges": positives}


def operating(y, s, threshold):
    chosen = s >= threshold
    tp, fp = int(np.sum(chosen & (y == 1))), int(np.sum(chosen & (y == 0)))
    fn, tn = int(y.sum()) - tp, int(np.sum(y == 0)) - fp
    n, count = len(y), tp + fp
    return {"threshold": threshold, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": tp / count if count else 1., "recall": tp / (tp + fn) if tp + fn else None,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.,
            "density": count / n, "sparsity": 1 - count / n,
            "false_discovery_fraction": fp / count if count else 0.,
            "edge_count_over_gold": count / int(y.sum()) if y.sum() else None,
            "normalized_structural_hamming_distance": (fp + fn) / n}


def full_curve(y, s):
    thresholds, tp, selected, precision, recall = curve_arrays(y, s)
    positive = int(y.sum())
    result = [operating(y, s, float(np.nextafter(float(s.max()), np.inf)))]
    for t, correct, count, p, r in zip(thresholds, tp, selected, precision, recall):
        fp, fn = int(count - correct), int(positive - correct)
        result.append({"threshold": float(t), "tp": int(correct), "fp": fp, "fn": fn,
            "tn": len(y) - positive - fp, "precision": float(p), "recall": float(r) if positive else None,
            "f1": float(2 * correct / (count + positive)) if count + positive else 0.,
            "density": float(count / len(y)), "sparsity": float(1 - count / len(y)),
            "false_discovery_fraction": float(1 - p), "edge_count_over_gold": float(count / positive) if positive else None,
            "normalized_structural_hamming_distance": (fp + fn) / len(y)})
    return result


def best(curve, constrained=False):
    available = [p for p in curve if not constrained or .15 <= p["density"] <= .30]
    return max(available, key=lambda p: (p["f1"], -p["density"], p["threshold"])) if available else None


def summarize(rows, name, thresholds=None, curves=None):
    points = [p for r in rows for p in r["points"][name]]
    y, s = arrays(points)
    curve = full_curve(y, s)
    if curves is not None:
        curves[name] = curve
    oracle = best(curve)
    near = [p for p in curve if p["f1"] >= oracle["f1"] - .02]
    per_case = [ranking(*arrays(r["points"][name])) for r in rows]
    result = {**ranking(y, s), "macro_trajectory_ap": float(np.mean([r["average_precision"] for r in per_case if r["average_precision"] is not None])),
        "n_trajectories": len(rows), "oracle_diagnostic_not_deployable": oracle,
        "oracle_with_density_15_30_diagnostic": best(curve, True),
        "within_002_of_oracle_f1_density_range": [min(p["density"] for p in near), max(p["density"] for p in near)],
        "by_family": {f: ranking(*arrays([p for r in rows if r["family"] == f for p in r["points"][name]])) for f in sorted({r["family"] for r in rows})}}
    if thresholds:
        result["frozen_dev_thresholds"] = {rule: operating(y, s, t) if t is not None else None for rule, t in thresholds.items()}
        result["fraction_trajectories_density_15_30"] = {
            rule: float(np.mean([.15 <= operating(*arrays(r["points"][name]), t)["density"] <= .30 for r in rows])) if t is not None else None
            for rule, t in thresholds.items()}
    return result


def paired_ap(rows, a, b, samples=2000):
    # Resample whole trajectories, stratified by operation family; never treat pairs as independent.
    rng = np.random.default_rng(2026101509)
    groups = [[i for i, r in enumerate(rows) if r["family"] == f] for f in sorted({r["family"] for r in rows})]
    cached = {name: [arrays(r["points"][name]) for r in rows] for name in (a, b)}
    def value(name, indexes):
        y = np.concatenate([cached[name][i][0] for i in indexes])
        s = np.concatenate([cached[name][i][1] for i in indexes])
        return ranking(y, s)["average_precision"]
    observed = value(b, range(len(rows))) - value(a, range(len(rows)))
    deltas = []
    for _ in range(samples):
        indexes = np.concatenate([rng.choice(g, len(g), replace=True) for g in groups])
        deltas.append(value(b, indexes) - value(a, indexes))
    return {"reference": a, "candidate": b, "micro_ap_gain": observed,
            "paired_family_stratified_trajectory_bootstrap_ci95": np.quantile(deltas, [.025, .975]).tolist(),
            "samples": samples, "confirmatory": False}


def read_rows(root, split):
    rows = []
    for shard in (0, 1):
        folder = root / f"{split}_{shard}"
        summary = json.loads((folder / "summary.json").read_text())
        if summary["status"] != "complete":
            raise ValueError(f"Incomplete worker: {folder}")
        rows.extend(json.loads(line) for line in (folder / "records.jsonl").open())
    expected = json.loads((root / f"{split}_cases.json").read_text())
    if len(rows) != len(expected) or {r["key"] for r in rows} != {r["key"] for r in expected}:
        raise ValueError("Missing or duplicated trajectories")
    for row in rows:
        if row["status"] != "complete":
            continue
        names = list(row["points"])
        reference = [(p["source"], p["target"], p.get("gold")) for p in row["points"][names[0]]]
        if any([(p["source"], p["target"], p.get("gold")) for p in row["points"][name]] != reference for name in names):
            raise ValueError("Unmatched edge pairs")
    return sorted(rows, key=lambda r: r["key"])


def analyze(root, split):
    rows = read_rows(root, split)
    failures = [r for r in rows if r["status"] != "complete"]
    if failures and split != "transfer":
        raise ValueError("Cannot omit failed labeled examples")
    rows = [r for r in rows if r["status"] == "complete"]
    names = list(rows[0]["points"])
    if split == "transfer":
        # No full gold graph exists for ATBench. Density is descriptive, not accuracy.
        frozen = json.loads((root / "DEV_THRESHOLDS.json").read_text())
        report = {"n_common": len(rows), "excluded": failures, "density": {}, "audit": {}, "selection_allowed": False}
        packet = json.loads((root / "AUDIT_BLIND.json").read_text())
        labels = {r["pair_id"]: r["label"] for r in json.loads((root / "AUDIT_ANNOTATIONS.json").read_text())["annotations"]}
        by_key = {r["key"]: r for r in rows}
        curves = {}
        for name in names:
            scores = np.array([p["score"] for r in rows for p in r["points"][name]])
            report["density"][name] = {rule: float(np.mean(scores >= t)) if t is not None else None for rule, t in frozen[name].items()}
            matched = []
            for pair in packet["sample"]:
                key = packet["contexts"][pair["context_id"]]["key"]
                if labels[pair["pair_id"]] not in (0, 1) or key not in by_key:
                    continue
                point = next(p for p in by_key[key]["points"][name] if p["source"] == pair["source"] and p["target"] == pair["target"])
                matched.append({**point, "gold": labels[pair["pair_id"]]})
            y, s = arrays(matched)
            curves[name] = full_curve(y, s)
            report["audit"][name] = {**ranking(y, s), "frozen_dev_thresholds": {rule: operating(y, s, t) if t is not None else None for rule, t in frozen[name].items()}}
        report["limitations"] = "Previously viewed, assistant-labeled 29 definite pairs with six positives, not independent or official edge gold. Audit PR curves are diagnostics only. Density measured on full graphs cannot establish correctness."
    else:
        curves = {}
        frozen = json.loads((root / "DEV_THRESHOLDS.json").read_text()) if split != "dev" else None
        report = {"metrics": {name: summarize(rows, name, frozen[name] if frozen else None, curves) for name in names}, "excluded": failures}
        if split == "dev":
            frozen = {name: {"max_dev_f1": best(curves[name])["threshold"],
                            "max_dev_f1_density_15_30": best(curves[name], True)["threshold"] if best(curves[name], True) else None}
                      for name in names}
            write(root / "DEV_THRESHOLDS.json", frozen)
        else:
            contrasts = [("H0_choice", "H0_noul"), ("short_neutral_choice", "short_neutral_noul"),
                         ("H0_choice", "short_choice"), ("H0_choice", "short_neutral_choice")]
            report["paired_ap"] = [paired_ap(rows, a, b) for a, b in contrasts]
            report["thresholds_sha256"] = digest(root / "DEV_THRESHOLDS.json")
    write(root / f"{split}_metrics.json", report)
    write(root / f"{split}_pr_curves.json", curves)
    print(json.dumps({"split": split, "status": "analyzed", "n_complete": len(rows), "n_excluded": len(failures)}), flush=True)


def dag_episode(number):
    """Random DAG dataflow, no benchmark text or semantic LLM annotation."""
    rng = random.Random(f"graph-ranking-new-dag:20261015:{number}")
    n = (8, 12, 16, 24)[number % 4]
    messages = [{"role": "user", "content": "Carry out the recorded artifact-processing workflow. Use only available inputs."}]
    gold, resources, notes = {}, [], []
    profile = ("acquire returns a fresh artifact for a named request. reshape consumes exactly its input artifact. "
               "fuse consumes all listed input artifacts. Each acquire/reshape/fuse output is a fresh artifact, not an alias. "
               "examine reads its input artifact and returns a fresh note. emit delivers the supplied note. "
               "heartbeat reports an independent clock reading and consumes no prior outputs.")
    def token():
        return f"v_{rng.getrandbits(64):016x}"
    for i in range(n):
        eid = f"E{i + 1}"
        mode = "acquire" if not resources else rng.choices(["acquire", "reshape", "fuse", "examine", "emit", "heartbeat"], [2, 4, 3, 2, 2 if notes else 0, 1])[0]
        if mode == "fuse" and len(resources) < 2:
            mode = "reshape"
        dependencies = []
        if mode == "acquire":
            arguments = {"request": token()}
        elif mode in ("reshape", "examine"):
            resource, producer = rng.choice(resources)
            arguments, dependencies = {"input": resource}, [producer]
        elif mode == "fuse":
            inputs = rng.sample(resources, min(len(resources), rng.choice([2, 3])))
            arguments, dependencies = {"inputs": [v for v, _ in inputs]}, [p for _, p in inputs]
        elif mode == "emit":
            note, producer = rng.choice(notes)
            arguments, dependencies = {"note": note}, [producer]
        else:
            arguments = {}
        if mode in ("acquire", "reshape", "fuse"):
            output = token()
            result = {"artifact": output, "status": "ok"}
            resources.append((output, eid))
        elif mode == "examine":
            note = "Observation " + token()
            result = {"note": note, "status": "ok"}
            notes.append((note, eid))
        else:
            result = {"status": "ok", "receipt": token()}
        messages.extend([{"role": "agent", "thought": "", "action": json.dumps({"name": mode, "arguments": arguments})},
                         {"role": "environment", "content": json.dumps(result)}])
        gold[eid] = dependencies
    case = {"key": f"fresh-dag:{number}", "family": f"dag_{n}", "profile": profile, "messages": messages, "gold_parents": gold}
    case["input_sha256"] = hashlib.sha256(json.dumps({"profile": profile, "messages": messages}, sort_keys=True).encode()).hexdigest()
    return case


def prepare(root, previous):
    from kev_link_prompt_search import PROMPTS, EVOLVE_FAMILIES, HOLDOUT_FAMILIES, episode
    from rjudge_kev_baseline import MODELS, SOURCE_COMMIT
    root.mkdir(parents=True, exist_ok=False)
    neutral = PROMPTS["short"].replace("Answer LINKED if", "A direct dependency exists if").replace("Otherwise answer NOT_LINKED.", "Otherwise no direct dependency exists.")
    variants = {
        "H0_choice": {"type": "choice", "instructions": PROMPTS["H0"]},
        "short_choice": {"type": "choice", "instructions": PROMPTS["short"]},
        "short_neutral_choice": {"type": "choice", "instructions": neutral},
        "H0_noul": {"type": "noul", "instructions": PROMPTS["H0"]},
        "short_neutral_noul": {"type": "noul", "instructions": neutral}}
    splits = {"dev": json.loads((previous / "evolve.json").read_text()),
              "legacy": json.loads((previous / "holdout_cases.json").read_text()),
              "fresh": [episode(f, i + 1000, "ranking_fresh") for f in EVOLVE_FAMILIES + HOLDOUT_FAMILIES for i in range(4)] + [dag_episode(i) for i in range(48)],
              "transfer": json.loads((previous / "transfer_cases.json").read_text())}
    for split, rows in splits.items():
        write(root / f"{split}_cases.json", rows)
    for filename in ("AUDIT_BLIND.json", "AUDIT_ANNOTATIONS.json", "AUDIT_FREEZE.json"):
        (root / filename).write_bytes((previous / filename).read_bytes())
    write(root / "PROTOCOL.json", {"created_utc": datetime.now(timezone.utc).isoformat(), "variants": variants,
        "model": MODELS["4b"], "source_commit": SOURCE_COMMIT,
        "primary_metric": "micro Average Precision: sum precision * recall increments, grouping equal-score pairs",
        "score_precision": "Native unrounded positive-option probabilities, not four-decimal API serialization; historical points recovered from calls.jsonl.",
        "secondary": ["macro trajectory AP", "ROC-AUC", "PR trapezoid AUC explicitly distinguished from AP", "family metrics"],
        "operating_points": "Per variant, maximize micro F1 on dev only; ties prefer fewer edges. Also freeze a density-15-30%-constrained dev optimum. No top-k or forced density at inference.",
        "curves": "Every distinct native score threshold, grouped ties; P/R/F1/density/sparsity/FP/FN/edge-count-to-gold/normalized SHD.",
        "oracle": "Same-split max F1 and density range within .02 of that maximum are diagnostics, never used to update frozen dev thresholds.",
        "contrasts": "H0 NOUL vs choice and neutral short NOUL vs choice have identical instructions and corresponding true/false criteria descriptions; original short choice retained to measure neutralization effect.",
        "budget": {s: len(r) for s, r in splits.items()}, "split_order": ["dev", "legacy", "fresh", "transfer"],
        "no_prompt_search": True, "no_safety_inference": True, "no_training": True, "no_truncation": True,
        "data_sha256": {s: digest(root / f"{s}_cases.json") for s in splits},
        "code_sha256": {p.name: digest(p) for p in Path(__file__).parent.glob("*.py")},
        "limitations": ["All prior dev and legacy results already viewed; only new trajectories are fresh.",
            "Fresh includes 48 instances of already studied templates and 48 new random-dataflow graphs. Not an independent natural-language edge benchmark.",
            "Same assistant authored data, labels and prompts; programmatic gold, no expert validation.",
            "ATBench has no official edge gold; old assistant audit is exploratory only.",
            "Density/sparsity are not correctness metrics and desirable density varies with true graph structure.",
            "AP depends on prevalence; compare variants only on matched pairs. Bootstrap intervals do not establish transfer to real-world trajectories."]})


def infer(root, split, shard):
    import os
    import signal
    import time
    import traceback
    import torch
    from kev.checkpoint import Checkpoint, LoadOptions
    from kev.model import ContextOverflow
    from kev.predictors import kernel_environment
    from kev.serve import Server
    from kev_hint_placement import Calls, readout_audit
    from kev_prompt_ablation import observations
    from rjudge_kev_baseline import MODELS, SOURCE_COMMIT

    protocol = json.loads((root / "PROTOCOL.json").read_text())
    if digest(root / f"{split}_cases.json") != protocol["data_sha256"][split]:
        raise ValueError("Frozen data changed")
    if split != "dev" and not (root / "DEV_THRESHOLDS.json").exists():
        raise ValueError("Freeze dev operating points before further inference")
    destination = root / f"{split}_{shard}"
    destination.mkdir(exist_ok=False)
    summary = {"status": "loading", "phase": split, "shard": shard, "attempted": 0,
        "protocol_sha256": digest(root / "PROTOCOL.json"), "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "thresholds_sha256": digest(root / "DEV_THRESHOLDS.json") if split != "dev" else None,
        "settings": {"bf16": True, "merge": True, "fused": True, "cuda_graphs": True,
                     "state_limit": 65536, "branch_limit": 8192, "row_limit": 73728, "truncation": False}}
    def save():
        write(destination / "summary.json", summary)
    def stop(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, stop)
    save()
    server, started = None, time.perf_counter()
    try:
        torch.manual_seed(0)
        spec = MODELS["4b"]
        source = Path(__import__("kev").__file__).parent.parent
        if (source / "SOURCE_COMMIT").read_text().strip() != SOURCE_COMMIT:
            raise ValueError("Source mismatch")
        ck = Checkpoint(f"{spec['repo']}@{spec['revision']}")
        if ck.weights_sha256() != spec["weights_sha"] or digest(ck.file("head.pt")) != spec["head_sha"]:
            raise ValueError("Weight mismatch")
        if ck.meta.base != spec["base"] or ck.meta.base_revision != spec["base_revision"]:
            raise ValueError("Base mismatch")
        tok, model = ck.load("cuda", LoadOptions(dtype=torch.bfloat16, merge=True, attn="sdpa", backend="torch", fused=True, cuda_graphs=True))
        server = Server(ck, tok, model, "cuda", truncate_states=False)
        calls = Calls(server, destination)
        summary.update(status="running", model=spec, temperature=model.head.temperature, environment=kernel_environment(model, "cuda"))
        cases = json.loads((root / f"{split}_cases.json").read_text())
        variants = protocol["variants"]
        with readout_audit(model, calls.captured):
            for index, case in enumerate(cases):
                if index % 2 != shard:
                    continue
                row = {"key": case["key"], "family": case.get("family", "ATBench"), "input_sha256": case["input_sha256"],
                       "status": "complete", "points": {name: [] for name in variants}, "cost": {name: 0 for name in variants}, "actions": []}
                try:
                    for eid, state, history in observations(case):
                        if not history:
                            continue
                        results = {}
                        order = list(variants)
                        random.Random(case["key"] + eid).shuffle(order)
                        for name in order:
                            spec = variants[name]
                            criteria = ({"NOT_LINKED": "No supported direct dependency.", "LINKED": "A supported direct dependency exists."}
                                if spec["type"] == "choice" else {"false": "No supported direct dependency.", "true": "A supported direct dependency exists."})
                            questions = {f"link_{sid}_{eid}": {"type": spec["type"], "instructions": spec["instructions"].format(source=sid, target=eid), "criteria": criteria} for sid in history}
                            result = calls.predict(case["key"], name, {"model": "kev-latest", "state": state, "questions": questions})
                            results[name] = result
                            native = {m["id"]: p[1] for m, p in zip(result["mapping"], result["native_probabilities"])}
                            for sid in history:
                                score = native[f"link_{sid}_{eid}"]
                                row["points"][name].append({"source": sid, "target": eid, "score": score,
                                    **({"gold": int(sid in case["gold_parents"][eid])} if "gold_parents" in case else {})})
                            row["cost"][name] += result["lengths"]["state"] + sum(result["lengths"]["branches"])
                        row["actions"].append({"id": eid, "state_sha256": hashlib.sha256(state.encode()).hexdigest(), "results": results})
                except ContextOverflow as error:
                    row = {"key": case["key"], "status": "rejected_context", "error": str(error)}
                    if split != "transfer":
                        raise
                except Exception:
                    row["status"], row["error"] = "failed", traceback.format_exc()
                    raise
                finally:
                    with (destination / "records.jsonl").open("a") as f:
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    summary["attempted"] += 1
                    save()
                print(json.dumps({"split": split, "shard": shard, "attempted": summary["attempted"], "status": row["status"]}), flush=True)
        summary["status"] = "complete"
    except KeyboardInterrupt:
        summary["status"] = "interrupted_by_user"
        raise
    except Exception:
        summary.update(status="failed", fatal_error=traceback.format_exc())
        raise
    finally:
        if server is not None:
            server.close()
        summary["seconds"] = time.perf_counter() - started
        save()


def historical(previous, output):
    output.mkdir(parents=True, exist_ok=True)
    report = {}
    for split, phase in (("dev", "evolve"), ("legacy", "holdout")):
        rows = []
        for shard in (0, 1):
            native = {}
            for line in (previous / f"{phase}_{shard}" / "calls.jsonl").open():
                call = json.loads(line)
                if "error" not in call:
                    native[call["call"]] = {m["id"]: p[1] for m, p in zip(call["mapping"], call["unrounded_probabilities"])}
            for line in (previous / f"{phase}_{shard}" / "records.jsonl").open():
                row = json.loads(line)
                for name, points in row["points"].items():
                    by_event = {a["id"]: native[a["results"][name]["call"]] for a in row["actions"] if name in a["results"]}
                    for point in points:
                        point["score"] = by_event[point["target"]][f"link_{point['source']}_{point['target']}"]
                rows.append(row)
        if any(r["status"] != "complete" for r in rows):
            raise ValueError("Incomplete historical synthetic run")
        curves = {}
        report[split] = {name: summarize(rows, name, curves=curves) for name in rows[0]["points"] if not name.startswith("H0_repeat")}
        write(output / f"historical_{split}_pr_curves.json", curves)
    write(output / "HISTORICAL_RANKING.json", report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "infer", "analyze", "historical"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--previous", type=Path)
    parser.add_argument("--split", choices=("dev", "legacy", "fresh", "transfer"))
    parser.add_argument("--shard", type=int, choices=(0, 1))
    args = parser.parse_args()
    if args.phase == "prepare":
        prepare(args.root, args.previous)
    elif args.phase == "historical":
        historical(args.previous, args.root)
    elif args.phase == "infer":
        infer(args.root, args.split, args.shard)
    else:
        analyze(args.root, args.split)


if __name__ == "__main__":
    main()
