"""Read-only reproduction of the archived NOUL safety sweep.

This adapter never imports historical inference scripts or their GPU dependencies.
It verifies the recorded cohort, inputs, arm coverage, trajectory aggregation and
reported numerical metrics before accepting an experiment as reproduced.
"""

import math
from pathlib import Path

from .io import read_json, read_jsonl, sha256, write_json
from .metrics import evaluate


def reproduce_safety(source, output):
    source, output = Path(source), Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite evidence: {output}")
    plan = read_json(source / "PLAN.json")
    original = read_json(source / "RESULTS.json")
    source_paths = [source / "PLAN.json", source / "RESULTS.json"]
    rows, summaries = [], []
    for shard in (0, 1):
        folder = source / f"safety_{shard}"
        summary = read_json(folder / "summary.json")
        if summary["status"] != "complete" or summary["plan_sha256"] != sha256(source / "PLAN.json"):
            raise ValueError("Incomplete worker or altered plan")
        summaries.append(summary)
        records = list(read_jsonl(folder / "records.jsonl"))
        if len(records) != summary["attempted"]:
            raise ValueError("Worker attempt count mismatch")
        rows.extend(records)
        source_paths.extend([folder / "summary.json", folder / "records.jsonl"])
    for field in ("model", "temperature", "environment", "settings", "threshold_sha256"):
        if summaries[0][field] != summaries[1][field]:
            raise ValueError(f"Worker mismatch: {field}")
    if len(rows) != len(plan["keys"]) or len({r["key"] for r in rows}) != len(rows) or {r["key"] for r in rows} != set(plan["keys"]):
        raise ValueError("Cohort mismatch")
    complete = [r for r in rows if r["status"] == "complete"]
    for row in complete:
        if set(row["scores"]) != set(plan["arms"]):
            raise ValueError("Arm coverage mismatch")
        for arm in plan["arms"]:
            if row["scores"][arm] != max(s["scores"][arm] for s in row["steps"]):
                raise ValueError("Incorrect max-over-actions aggregation")
    reports, differences = {}, []
    scopes = {"all_100_exploratory": set(plan["keys"]),
              "previous_evaluation_80_diagnostic": set(plan["evaluation_keys"])}
    for scope, keys in scopes.items():
        selected = [r for r in complete if r["key"] in keys]
        reports[scope] = {}
        for arm in plan["arms"]:
            result = evaluate([r["label"] for r in selected], [r["scores"][arm] for r in selected])
            old = original["scopes"][scope]["metrics"][arm]
            compare = {**result["operating_point"], "average_precision": result["average_precision"], "roc_auc": result["roc_auc"]}
            for name in ("tp", "fp", "tn", "fn", "precision", "recall", "specificity", "accuracy", "f1", "balanced_accuracy", "average_precision", "roc_auc"):
                if compare[name] is None or not math.isclose(compare[name], old[name], rel_tol=0, abs_tol=1e-12):
                    differences.append({"scope": scope, "arm": arm, "metric": name, "old": old[name], "new": compare[name]})
            reports[scope][arm] = result
    status = "reproduced" if not differences else "mismatch"
    write_json(output, {"status": status, "metric_version": 2,
        "source_manifest": [{"path": str(p.resolve()), "sha256": sha256(p)} for p in source_paths],
        "n_attempted": len(rows), "n_common": len(complete), "differences": differences,
        "excluded": [{k: r.get(k) for k in ("key", "label", "status", "error")} for r in rows if r["status"] != "complete"],
        "scopes": reports,
        "scope_of_verification": "Point metrics and stored trajectory aggregation only; does not validate gold labels, model inference, all historical runs or uncertainty estimates."})
    if differences:
        raise ValueError(f"Metric mismatches recorded in {output}")
    return {"status": status, "n_common": len(complete), "n_arms": len(plan["arms"]), "output": str(output)}
