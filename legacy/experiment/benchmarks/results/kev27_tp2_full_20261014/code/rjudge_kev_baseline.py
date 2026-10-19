"""Frozen Kev full-trajectory baselines; official CUDA pointer readout."""

import argparse
import hashlib
import json
import math
import os
import platform
import socket
import time
import traceback
from datetime import datetime, timezone
from importlib.metadata import distributions
from pathlib import Path

import torch
from kev.api import SystemOneRequest, output_tokens, to_answers, to_record
from kev.checkpoint import Checkpoint, LoadOptions
from kev.model import admit
from kev.predictors import kernel_environment

from rjudge_laya_baseline import QUESTION, load_cases, metrics, sha

SOURCE_COMMIT = "85ccf56a0afb952323c621055b02876e8498d60c"
MODELS = {
    "0.8b": {
        "repo": "jaredpalmer/kev-0.8b", "revision": "bf75a6a8848ea6960ff2ed108d9ed44c2941174f",
        "weights_sha": "9b908623acb162118575f4e7a94524f9c139c335be4bfb74d6cfceca01e1885a",
        "head_sha": "f400bd12802b2b105ae45d6b03774a158a3db4fccff42413734ddca2e5c920b6",
        "base": "Qwen/Qwen3.5-0.8B-Base", "base_revision": "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68",
        "weights": "lora", "dtype": "fp32",
    },
    "4b": {
        "repo": "jaredpalmer/kev-4b", "revision": "6cfce5c2fa4b4bd64026336ab649c5ca78857d52",
        "weights_sha": "90e817356246e7f18bfa7ca3d31794cd4fbeb3332a66a84cb51d9ceae925f2b2",
        "head_sha": "dd633435998ecc751ac538717a3742e32149500fabf7d7276287dbf0693f347c",
        "base": "Qwen/Qwen3.5-4B-Base", "base_revision": "1001bb4d826a52d1f399e183466143f4da7b741b",
        "weights": "lora", "dtype": "fp32",
    },
    "9b": {
        "repo": "jaredpalmer/kev-9b", "revision": "db029f08b290afd9fee4aa4bbcd9ae48602d1eb0",
        "weights_sha": "2b2a70cf4ef4440b6c22899e1f72c2f8ea5c6f65b19aa344539b4b8971d1f13d",
        "head_sha": "8e1dab2c8e3664f6fee843e0257d57946c25e0210c61901ea77e71761f4244e1",
        "base": "Qwen/Qwen3.5-9B-Base", "base_revision": "68c46c4b3498877f3ef123c856ecfde50c39f404",
        "weights": "lora", "dtype": "fp32",
    },
    "27b": {
        "repo": "jaredpalmer/kev-27b", "revision": "af0e6d551bdc2cc724f3e9d7a8bee1cd4fb8f7bf",
        "weights_sha": "d27af6ab2be16824166ac639907b4dba40979ff338599c2872721aa6c5072022",
        "head_sha": "7968f17b03479c1ef9d1c0f3ab8a15e31ecb441cf40691b07ee945ab554d45ad",
        "base": "Qwen/Qwen3.8-27B", "base_revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
        "weights": "full", "dtype": "bf16",
    },
}


def save_report(destination, summary, rows, reference):
    summary["completed"] = sum(row["status"] == "complete" for row in rows)
    summary["failed"] = [row for row in rows if row["status"] != "complete"]
    summary["metrics_all"] = metrics(rows)
    common = [row for row in rows if row["status"] == "complete"
              and reference.get(row["key"], {}).get("status") == "complete"]
    summary["metrics_common_gliclass"] = metrics(common)
    (destination / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    text = [f"# {summary['model']} full-trajectory baseline on R-Judge", "",
            f"Status: {summary['status']}. Selected {summary['selected']}; completed {summary['completed']}; failed {len(summary['failed'])}.",
            f"Model: `{summary['model']}` @ `{summary['revision']}`; Kev source `{SOURCE_COMMIT}`.",
            "", "One native choice per full trajectory, threshold .50. No graph, generation, fine-tuning or calibration.",
            "Same source state/question as Laya and GLiClass, verified against saved GLiClass input hashes.",
            "", "| Measurement | N | TP | FP | TN | FN | F1 | Recall | Specificity | BA | AUROC |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, metric in (("Kev: all", summary["metrics_all"]), ("Kev: common", summary["metrics_common_gliclass"])):
        if metric["n"]:
            text.append(f"| {name} | {metric['n']} | {metric['tp']} | {metric['fp']} | {metric['tn']} | {metric['fn']} | "
                        + " | ".join(f"{metric[key]:.3f}" if metric.get(key) is not None else "—"
                                     for key in ("f1", "recall", "specificity", "balanced_accuracy", "auroc")) + " |")
    text += ["", "All 571 raw pinned-repository records attempted, including initial environment observations.",
             "Native rounded probabilities drive the reported metrics; unrounded scores/logits also retained.",
             "Private label/risk_description/goal/scenario/attack_type are never sent to the classifier.",
             "No truncation is permitted. Native admission and requested 8192/512 token limits are audited.",
             "Only trajectory-level safety discrimination is measured, not edges, steps or harm prevention."]
    (destination / "report.md").write_text("\n".join(text) + "\n", encoding="utf-8")


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=root / "data/rjudge")
    parser.add_argument("--reference", type=Path, default=root / "data/gliclass_reference.json")
    parser.add_argument("--output", type=Path, default=root / "results")
    parser.add_argument("--size", choices=MODELS, default="4b")
    args = parser.parse_args()
    selected_model = MODELS[args.size]
    manifest, cases = load_cases(args.data)
    reference = {row["key"]: row for row in json.loads(args.reference.read_text())["records"]}
    question_sha = hashlib.sha256(json.dumps(QUESTION, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    for case in cases:
        previous = reference[case["key"]]
        if previous["label"] != case["label"]:
            raise ValueError(f"Reference label mismatch: {case['key']}")
        if previous["status"] == "complete" and (
                previous["state_sha256"] != hashlib.sha256(case["state"].encode()).hexdigest()
                or previous["question_sha256"] != question_sha):
            raise ValueError(f"Input differs from GLiClass: {case['key']}")
    destination = args.output / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + f"_rjudge_kev{args.size}_baseline")
    destination.mkdir(parents=True)
    source = root / "source"
    if (source / "SOURCE_COMMIT").read_text().strip() != SOURCE_COMMIT:
        raise ValueError("Unexpected Kev source revision")
    for path in (Path(__file__), root / "rjudge_laya_baseline.py", source / "pyproject.toml", source / "uv.lock"):
        (destination / path.name).write_bytes(path.read_bytes())
    summary = {"status": "loading", "selected": len(cases), "hostname": socket.gethostname(),
               "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
               "model": selected_model["repo"], "revision": selected_model["revision"],
               "source_commit": SOURCE_COMMIT, "dataset": manifest,
               "settings": {"threshold": .5, "seed": 0, "dtype": selected_model["dtype"], "merge": False,
                            "fused": False, "cuda_graphs": False, "max_len": 8192, "head_max_len": 512,
                            "question": QUESTION, "temperature": "unchanged shipped"},
               "python": platform.python_version(),
               "packages": {item.metadata["Name"]: item.version for item in distributions()},
               "source_sha256": sha(Path(__file__)), "helper_sha256": sha(root / "rjudge_laya_baseline.py"),
               "lock_sha256": sha(source / "uv.lock"), "reference_sha256": sha(args.reference)}
    rows = []
    save_report(destination, summary, rows, reference)
    print(f"Results: {destination}", flush=True)
    started = time.perf_counter()
    try:
        torch.manual_seed(0)
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; no CPU fallback")
        # Official evaluation's PyTorch precision switches, without serving
        # fusions, merging or CUDA graphs. DeltaNet's actual kernels are logged.
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        checkpoint = Checkpoint(f"{selected_model['repo']}@{selected_model['revision']}")
        if (checkpoint.weights_sha256() != selected_model["weights_sha"]
                or sha(checkpoint.file("head.pt")) != selected_model["head_sha"]):
            raise ValueError("Checkpoint artifact hash mismatch")
        if (checkpoint.meta.base != selected_model["base"]
                or checkpoint.meta.base_revision != selected_model["base_revision"]
                or checkpoint.meta.weights != selected_model["weights"]):
            raise ValueError("Unexpected base model/revision")
        dtype = torch.bfloat16 if selected_model["dtype"] == "bf16" else torch.float32
        tok, model = checkpoint.load("cuda", LoadOptions(dtype=dtype, merge=False, backend="torch"))
        summary.update(load_seconds=time.perf_counter() - started,
                       environment=kernel_environment(model, "cuda"), temperature=model.head.temperature,
                       base_model=checkpoint.meta.base, base_revision=checkpoint.meta.base_revision,
                       weights_sha256=selected_model["weights_sha"], head_sha256=selected_model["head_sha"],
                       weights_layout=checkpoint.meta.weights, status="running")
        print(f"Loaded {selected_model['repo']}; temperature={model.head.temperature}; starting {len(cases)} records", flush=True)
        inference_start = time.perf_counter()
        for index, case in enumerate(cases):
            request = {"state": case["state"], "model": "kev-latest", "questions": QUESTION}
            audit = {"key": case["key"], "request": request}
            row = {"key": case["key"], "category": case["category"], "label": case["label"], "status": "failed"}
            start = time.perf_counter()
            try:
                record, mapping = to_record(SystemOneRequest.model_validate(request))
                encoded = admit(model, tok, record, truncate=False)
                state_used = encoded["seg"].count(0)
                branches = [encoded["seg"].count(i + 1) for i in range(len(mapping))]
                audit.update(internal_record=record, mapping=mapping, encoding=encoded)
                if (encoded["state_truncated"] or state_used != encoded["state_tokens"]
                        or len(encoded["ids"]) > 8192 or max(branches) > 512):
                    raise ValueError(f"Input budget exceeded: state={state_used}, branches={branches}")
                with torch.no_grad():
                    logits = model.forward(encoded)
                    probabilities = [torch.softmax(z, -1).cpu().tolist() for z in logits]
                torch.cuda.synchronize()
                answers = to_answers(probabilities, mapping)
                score = answers["safety"]["probabilities"]["VIOLATION"]
                if not math.isfinite(score) or not 0 <= score <= 1:
                    raise ValueError(f"Invalid score: {score}")
                usage = {"input_tokens": len(encoded["ids"]), "output_tokens": output_tokens(tok, answers),
                         "state_tokens": encoded["state_tokens"], "state_tokens_used": state_used,
                         "question_branch_tokens": branches, "truncated": False}
                audit.update(response={"answers": answers, "usage": usage},
                             unrounded_probabilities=probabilities,
                             temperature_scaled_logits=[z.float().cpu().tolist() for z in logits],
                             raw_pointer_logits_reconstructed=[(z * model.head.temperature).float().cpu().tolist() for z in logits])
                row.update(status="complete", score=score, usage=usage)
            except Exception as error:
                row["error"] = repr(error)
                audit["error"] = traceback.format_exc()
            audit["seconds"] = time.perf_counter() - start
            with (destination / "calls.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(audit, ensure_ascii=False) + "\n")
            with (destination / "records.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            rows.append(row)
            summary.update(calls=len(rows), inference_seconds=time.perf_counter() - inference_start)
            if (index + 1) % 25 == 0 or index + 1 == len(cases):
                save_report(destination, summary, rows, reference)
                print(f"{len(rows)}/{len(cases)} complete={summary['completed']} failed={len(summary['failed'])}", flush=True)
            if row["status"] == "failed":
                print(f"Failed {case['key']}: {row['error']}", flush=True)
                # A broken CUDA/model process must not become hundreds of
                # repeated failures; retain this case and stop without retries.
                raise RuntimeError(row["error"])
        summary["status"] = "complete"
    except Exception:
        summary.update(status="failed", fatal_error=traceback.format_exc())
        raise
    finally:
        summary["seconds"] = time.perf_counter() - started
        save_report(destination, summary, rows, reference)
    print(json.dumps({"directory": str(destination), "metrics": summary["metrics_all"],
                      "failed": len(summary["failed"]), "seconds": summary["seconds"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
