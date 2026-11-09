"""Evaluate a candidate judge prompt on a gold set.

Gold labels: 0=safe, 1=unsafe, 2=uncertain. Predictions are mapped the same way.
The script reports 3-class macro F1, binary accuracy (uncertain counts as wrong),
unsafe recall/precision, coverage and token cost. It never reads labels from the
model input; labels are only used for scoring.

Example:
    python eval_prompt.py \
      --gold /data/gold_cases.jsonl --split dev \
      --system-file /scripts/prompts/v1.txt \
      --output /data/eval_v1_dev.jsonl
"""
import argparse
import asyncio
import json
import statistics
import time
from collections import Counter
from pathlib import Path

import httpx

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None

import sys

sys.path.insert(0, "/scripts")
from annotate import SCHEMA  # noqa: E402

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


async def one(client, sem, case, system, args):
    async with sem:
        user_text = json.dumps(case["state"], ensure_ascii=False)
        payload = {
            "model": args.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user_text},
            ],
            "temperature": args.temperature,
            "top_p": 1.0,
            "max_tokens": args.max_tokens,
            "seed": args.seed,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "annotation", "strict": True, "schema": SCHEMA},
            },
            "chat_template_kwargs": {"enable_thinking": False},
        }
        started = time.monotonic()
        last = None
        for attempt in range(1, args.retries + 1):
            try:
                response = await client.post(
                    f"{args.base_url.rstrip('/')}/v1/chat/completions",
                    json=payload,
                    timeout=args.timeout,
                )
                if response.status_code != 200:
                    last = f"HTTP {response.status_code}: {response.text[:300]}"
                    await asyncio.sleep(min(2 ** attempt, 10))
                    continue
                body = response.json()
                content = body["choices"][0]["message"]["content"]
                try:
                    parsed = json.loads(content)
                except json.JSONDecodeError:
                    start = content.find("{")
                    end = content.rfind("}")
                    if start < 0 or end <= start:
                        raise
                    parsed = json.loads(content[start : end + 1])
                verdict = parsed.get("verdict")
                if verdict not in LABEL:
                    raise ValueError(f"bad verdict {verdict!r}")
                usage = body.get("usage") or {}
                return {
                    "key": case["key"],
                    "gold": case["gold"],
                    "pred": LABEL[verdict],
                    "verdict": verdict,
                    "facets": parsed.get("facets"),
                    "reason": parsed.get("reason"),
                    "status": "ok",
                    "tokens": (usage.get("prompt_tokens") or 0) + (usage.get("completion_tokens") or 0),
                    "usage": usage,
                    "latency_ms": round((time.monotonic() - started) * 1000, 2),
                    "attempt": attempt,
                }
            except Exception as error:
                last = f"{type(error).__name__}: {error}"
                await asyncio.sleep(min(2 ** attempt, 10))
        return {
            "key": case["key"],
            "gold": case["gold"],
            "pred": None,
            "status": "error",
            "error": last,
            "latency_ms": round((time.monotonic() - started) * 1000, 2),
        }


async def run(args):
    cases = [json.loads(line) for line in args.gold.open() if line.strip()]
    cases = [c for c in cases if c["split"] == args.split]
    system = args.system_file.read_text()
    sem = asyncio.Semaphore(args.concurrency)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    async with httpx.AsyncClient() as client:
        tasks = [one(client, sem, case, system, args) for case in cases]
        if tqdm is None:
            rows = await asyncio.gather(*tasks)
        else:
            rows = []
            for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="eval"):
                rows.append(await coro)
    with args.output.open("w") as stream:
        for row in sorted(rows, key=lambda r: r["key"]):
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    result = metrics(rows)
    result["split"] = args.split
    result["system_file"] = str(args.system_file)
    result["temperature"] = args.temperature
    print(json.dumps(result, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--split", choices=["dev", "heldout"], default="dev")
    parser.add_argument("--system-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:18000")
    parser.add_argument("--model", default="qwen3.8-flash-next")
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=768)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20261109)
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
