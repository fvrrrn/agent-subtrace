"""Score a judge prompt against the gold set.

Gold labels: 0=safe, 1=unsafe, 2=uncertain; predictions are mapped the same way.
Reports 3-class macro F1, binary accuracy (uncertain counts as wrong), unsafe
recall/precision, coverage and token cost. Labels are used for scoring only and
never reach the judge: the model sees the same `state` as in `annotate.py`.

    uv run python scripts/judges/eval.py \
        --gold data/gold_cases.jsonl --split dev \
        --system-file scripts/dataset/flash_next/prompt.txt \
        --model qwen/qwen3-235b-a22b-2507 \
        --output data/eval_qwen3-235b_dev.jsonl

The run file it writes is what `aggregate_runs.py` and `diff_runs.py` read.
"""
import argparse
import asyncio
import json
import statistics
import time
from collections import Counter
from pathlib import Path

from openai import AsyncOpenAI
from tqdm import tqdm

from annotate import API_KEY_ENVS, BASE_URL_ENVS, DEFAULT_BASE_URL, env_value, request_annotation

LABEL = {"safe": 0, "unsafe": 1, "uncertain": 2}


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
    gold = [r["gold"] for r in rows if r["status"] == "ok"]
    pred = [r["pred"] for r in rows if r["status"] == "ok"]
    if not gold:
        return {"n": 0}
    covered = [(g, p) for g, p in zip(gold, pred) if p != 2]
    binary = [(g, p) for g, p in zip(gold, pred) if g != 2]
    unsafe = [(g, p) for g, p in zip(gold, pred) if g == 1]
    safe = [(g, p) for g, p in zip(gold, pred) if g == 0]
    predicted_unsafe = [(g, p) for g, p in zip(gold, pred) if p == 1]
    cost = [r["tokens"] for r in rows if r["status"] == "ok" and r.get("tokens") is not None]
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
        "cost_total": sum(cost) if cost else None,
    }


async def one(client, sem, case, args):
    async with sem:
        user_text = json.dumps(case["state"], ensure_ascii=False)
        messages = [
            {"role": "system", "content": args.system},
            {"role": "user", "content": user_text},
        ]
        started = time.monotonic()
        parsed, completion, mode, error = await request_annotation(client, args, messages)
        latency = round((time.monotonic() - started) * 1000, 2)

        if parsed is None:
            return {"key": case["key"], "gold": case["gold"], "pred": None, "status": "error", "error": error, "latency_ms": latency}

        usage = completion.usage.model_dump() if completion.usage else {}
        return {
            "key": case["key"],
            "gold": case["gold"],
            "pred": LABEL[parsed["verdict"]],
            "verdict": parsed["verdict"],
            "facets": parsed.get("facets"),
            "reason": parsed.get("reason"),
            "status": "ok",
            "response_format": mode,
            "tokens": (usage.get("prompt_tokens") or 0) + (usage.get("completion_tokens") or 0),
            "usage": usage,
            "latency_ms": latency,
        }


async def run(args):
    cases = [json.loads(line) for line in args.gold.open(encoding="utf-8") if line.strip()]
    cases = [case for case in cases if case["split"] == args.split]
    sem = asyncio.Semaphore(args.concurrency)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    client = AsyncOpenAI(
        base_url=args.base_url,
        api_key=args.api_key or "EMPTY",
        timeout=args.timeout,
        max_retries=args.retries,
    )
    try:
        tasks = [one(client, sem, case, args) for case in cases]
        rows = []
        for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="eval", unit="row"):
            rows.append(await coro)
    finally:
        await client.close()

    with args.output.open("w", encoding="utf-8") as stream:
        for row in sorted(rows, key=lambda item: item["key"]):
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    result = metrics(rows)
    result["split"] = args.split
    result["model"] = args.model
    result["system_file"] = str(args.system_file)
    result["temperature"] = args.temperature
    print(json.dumps(result, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--split", choices=["dev", "heldout"], default="dev")
    parser.add_argument("--system-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", default=env_value(BASE_URL_ENVS) or DEFAULT_BASE_URL)
    parser.add_argument("--api-key", default=env_value(API_KEY_ENVS))
    parser.add_argument("--json-mode", choices=["auto", "schema", "object", "none"], default="auto")
    parser.add_argument("--extra-body", type=json.loads, default=None)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=768)
    parser.add_argument("--seed", type=int, default=20261109)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--parse-retries", type=int, default=2)
    args = parser.parse_args()

    args.modes = ["schema", "object", "none"] if args.json_mode == "auto" else [args.json_mode]
    args.system = args.system_file.read_text(encoding="utf-8")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
