"""Frozen Clef ATBench replay using the official Transformers joint-schema loader."""

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import signal
import sys
import time
import traceback

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from kev_hint_placement import ContextOverflow, evaluate, load_cases, LINK, RELATIONS, SAFETY, HINT
from kev_hint_full_report import aggregate_full

REVISION = "8b2e5fd17c09fd49bc2880b805d5323ec7ff4ff3"
STATE_LIMIT = 65536
TOTAL_LIMIT = 131072


class Calls:
    def __init__(self, model, processor, release, destination):
        self.model, self.processor, self.release = model, processor, release
        self.destination, self.count = destination, 0

    def validate(self, request):
        tokenizer = self.processor.tokenizer
        # The graph is extra schema/context, not part of the chronological history budget.
        state = request["state"].split("\n\n<dependency_hint>\n", 1)[0]
        state_tokens = len(tokenizer(state, add_special_tokens=False).input_ids)
        if state_tokens > STATE_LIMIT:
            raise ContextOverflow(f"History has {state_tokens} tokens; limit {STATE_LIMIT}")
        # Encode above the allowed budget first, so the official slicer cannot hide overflow.
        encoded = self.release.encode_record(tokenizer, request, max_length=1048576,
                                             processor=self.processor)
        total = len(encoded.input_ids)
        if total > TOTAL_LIMIT:
            raise ContextOverflow(f"Complete joint-schema input has {total} tokens; limit {TOTAL_LIMIT}")
        return encoded, {"history": state_tokens, "total": total,
                         "question_count": len(encoded.questions)}

    def predict(self, key, stage, request):
        request = {**request, "model": "Cloudflare/clef-flash"}
        audit = {"call": self.count, "key": key, "stage": stage, "request": request}
        self.count += 1
        try:
            started = time.perf_counter()
            encoded, lengths = self.validate(request)
            audit.update(lengths=lengths, input_ids=encoded.input_ids,
                         questions=[vars(q) for q in encoded.questions])
            batch = self.release.collate_records([encoded], self.processor.tokenizer.pad_token_id,
                                                 torch.device("cuda"))
            audit["encoding_seconds"] = time.perf_counter() - started
            torch.cuda.synchronize()
            started = time.perf_counter()
            with torch.inference_mode(), sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
                logits = self.model(batch)[0]
            torch.cuda.synchronize()
            seconds = time.perf_counter() - started
            raw = [row.float().cpu().tolist() for row in logits]
            probabilities = [row.float().softmax(-1).cpu().tolist() for row in logits]
            answers = {q.question_id: self.release.systemone_answer(
                request["questions"][q.question_id], dict(zip(q.option_ids, probs)))
                for q, probs in zip(encoded.questions, probabilities)}
            stats = {"prefix_cache_hit": False, "input_tokens": lengths["total"],
                     "peak_memory_bytes": torch.cuda.max_memory_allocated()}
            audit.update(seconds=seconds, raw_logits=raw, unrounded_probabilities=probabilities,
                         response={"answers": answers, "stats": stats})
            return {"call": audit["call"], "seconds": seconds, "stats": stats,
                    "lengths": lengths, "answers": answers}
        except Exception:
            audit["error"] = traceback.format_exc()
            raise
        finally:
            with (self.destination / "calls.jsonl").open("a") as handle:
                handle.write(json.dumps(audit, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--shard", type=int, choices=range(3))
    parser.add_argument("--aggregate", action="store_true")
    args = parser.parse_args()
    if args.aggregate:
        aggregate_full(args.output)
        return
    if args.data is None or args.model_path is None or (not args.baseline and args.shard is None):
        parser.error("Need model, data, and baseline or shard")
    if args.model_path.name != REVISION:
        raise ValueError("Unexpected Clef revision")
    sys.path.insert(0, str(args.model_path))
    import joint_schema_model as release

    manifest, cases = load_cases(args.data)
    destination = args.output / ("baseline" if args.baseline else f"shard_{args.shard}")
    destination.mkdir(parents=True, exist_ok=False)
    prompts = {"link": LINK, "relations": RELATIONS, "safety": SAFETY, "hint": HINT}
    prompt_hash = hashlib.sha256(json.dumps(prompts, sort_keys=True).encode()).hexdigest()
    thresholds = json.loads((args.output / "thresholds.json").read_text())
    if thresholds["prompts_sha256"] != prompt_hash:
        raise ValueError("Frozen prompts mismatch")
    summary = {"status": "loading", "created_utc": datetime.now(timezone.utc).isoformat(),
               "dataset": manifest, "selection": [{"key": c["key"], "label": c["label"]} for c in cases],
               "prompts": prompts, "prompts_sha256": prompt_hash, "threshold_selection": thresholds,
               "baseline": args.baseline, "shard": args.shard, "shards": 3, "attempted": 0,
               "model": {"repo": "Cloudflare/clef-flash", "revision": REVISION}, "temperature": None,
               "settings": {"state_limit": STATE_LIMIT, "joint_input_limit": TOTAL_LIMIT,
                            "link_threshold": .3, "safety_threshold": .5, "truncation": False,
                            "aggregation": "max over causal actions", "dtype": "bf16",
                            "attention": "torch SDPA flash/efficient only", "cache": False,
                            "question_semantics": "joint schema, not Kev-isolated branches"},
               "environment": {p: version(p) for p in ("torch", "transformers",
                                                       "flash-linear-attention", "fla-core")}}
    def save():
        (destination / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    save()
    def stop(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, stop)
    started = time.perf_counter()
    try:
        torch.manual_seed(0)
        model, processor = release.load_release_model(args.model_path, device="cuda",
                                                      attn_implementation="sdpa")
        summary["load_seconds"] = time.perf_counter() - started
        calls = Calls(model, processor, release, destination)
        # Synthetic preflight only; no benchmark labels or tuning.
        for length in (64, 64000):
            calls.predict("WARMUP", "context_preflight", {
                "state": "public " * length, "questions": {"safety": SAFETY}})
        summary["status"] = "running"
        save()
        run_start = time.perf_counter()
        for index, case in enumerate(cases):
            if not args.baseline and index % 3 != args.shard:
                continue
            try:
                row = evaluate(calls, case, index, baseline=args.baseline)
            except ContextOverflow as error:
                row = {"key": case["key"], "label": case["label"], "status": "rejected_context",
                       "error": str(error)}
            except Exception:
                row = {"key": case["key"], "label": case["label"], "status": "failed",
                       "error": traceback.format_exc()}
                with (destination / "records.jsonl").open("a") as handle:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                summary["attempted"] += 1
                raise
            with (destination / "records.jsonl").open("a") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            summary["attempted"] += 1
            save()
            print(json.dumps({"worker": destination.name, "key": row["key"], "status": row["status"]}), flush=True)
        summary.update(status="complete", inference_seconds=time.perf_counter() - run_start)
    except KeyboardInterrupt:
        summary["status"] = "interrupted_by_user"
        raise
    except Exception:
        summary.update(status="failed", fatal_error=traceback.format_exc())
        raise
    finally:
        summary["seconds"] = time.perf_counter() - started
        save()


if __name__ == "__main__":
    main()
