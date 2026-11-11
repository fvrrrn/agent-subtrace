r"""Annotate action candidates with any OpenAI-compatible chat-completions judge.

The judge is whatever serves `POST {base_url}/chat/completions`: OpenRouter,
vLLM, Ollama, or a vendor API. The prompt comes from `--system-file`, the
candidate's visible state goes as a single JSON user message, and every row is
written as one JSON object that `scripts/dataset/build_dataset.py` can join.

The record shape is the one the released labels were produced with, so two
annotation files can be diffed field by field.

    uv run python scripts/judges/annotate.py \
        --input data/candidates.jsonl \
        --output data/annotations.jsonl \
        --model qwen/qwen3-235b-a22b-2507 \
        --system-file scripts/dataset/flash_next/prompt.txt

Credentials come from the environment, with `.env` loaded from the working
directory upwards: `JUDGE_API_KEY` / `OPENAI_API_KEY` / `OPENROUTER_API_KEY`,
and `JUDGE_BASE_URL` / `OPENAI_BASE_URL` / `OPENROUTER_BASE_URL`.

Structured output degrades on demand: `json_schema` first, then `json_object`,
then plain text with JSON extraction. `--json-mode` pins one of them.

Runs resume: rows already marked `ok` or `too_long` are skipped, `error` rows
are retried on the next pass.
"""
import argparse
import asyncio
import hashlib
import json
import os
import time
from collections import Counter
from pathlib import Path

from dotenv import find_dotenv, load_dotenv
from openai import AsyncOpenAI, BadRequestError
from tqdm import tqdm

FACETS = [
    "data_exposure",
    "destructive",
    "financial",
    "credential_identity",
    "authorization_scope",
    "injection_hierarchy",
    "code_execution",
    "deception_social",
    "physical_cyber_harm",
    "policy_legal",
]

SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["safe", "unsafe", "uncertain"]},
        "facets": {"type": "array", "items": {"type": "string", "enum": FACETS}, "maxItems": 3},
        "severity": {"type": "string", "enum": ["low", "medium", "high", "unknown"]},
        "reversibility": {"type": "string", "enum": ["reversible", "irreversible", "unknown"]},
        "actor": {"type": "string", "enum": ["user", "third_party", "injected", "unknown"]},
        "authorization": {"type": "string", "enum": ["explicit", "ambiguous", "absent", "unknown"]},
        "evidence_sufficiency": {"type": "string", "enum": ["sufficient", "insufficient", "unknown"]},
        "evidence": {"type": "array", "items": {"type": "string", "maxLength": 200}, "maxItems": 3},
        "reason": {"type": "string", "maxLength": 400},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "required": [
        "verdict",
        "facets",
        "severity",
        "reversibility",
        "actor",
        "authorization",
        "evidence_sufficiency",
        "evidence",
        "reason",
        "confidence",
    ],
    "additionalProperties": False,
}

REQUIRED_FIELDS = SCHEMA["required"]

RESPONSE_FORMATS = {
    "schema": {
        "type": "json_schema",
        "json_schema": {"name": "annotation", "strict": True, "schema": SCHEMA},
    },
    "object": {"type": "json_object"},
}

API_KEY_ENVS = ("JUDGE_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY")
BASE_URL_ENVS = ("JUDGE_BASE_URL", "OPENAI_BASE_URL", "OPENROUTER_BASE_URL")
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"


def digest(value):
    if not isinstance(value, str):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(value.encode()).hexdigest()


def env_value(names):
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return None


def sources_of(row):
    result = set()
    for item in row.get("provenance") or []:
        if isinstance(item, dict) and item.get("source"):
            result.add(str(item["source"]))
    if not result and row.get("sources"):
        result.update(str(s) for s in row["sources"])
    return sorted(result)


def visible_state(row):
    state = row.get("state")
    if isinstance(state, str):
        state = json.loads(state)
    if not isinstance(state, dict):
        raise ValueError("state is not an object")
    action = state.get("proposed_action")
    if isinstance(action, dict) and "tool_result" in action:
        raise ValueError("proposed_action contains tool_result")
    return state


def extract_json(content):
    if not content:
        return None
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass
    start, end = content.find("{"), content.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        return json.loads(content[start : end + 1])
    except json.JSONDecodeError:
        return None


def schema_problems(parsed):
    if not isinstance(parsed, dict):
        return ["not_an_object"]
    problems = [f"missing:{name}" for name in REQUIRED_FIELDS if name not in parsed]
    if parsed.get("verdict") not in ("safe", "unsafe", "uncertain"):
        problems.append("bad_verdict")
    facets = parsed.get("facets")
    if not isinstance(facets, list):
        problems.append("bad_facets")
    elif any(facet not in FACETS for facet in facets):
        problems.append("unknown_facet")
    return problems


def validate(parsed, user_text):
    warnings = []
    if parsed.get("verdict") == "unsafe" and not parsed.get("facets"):
        warnings.append("unsafe_without_facets")
    if parsed.get("verdict") == "safe" and parsed.get("facets"):
        warnings.append("safe_with_facets")
    evidence_valid = True
    for quote in parsed.get("evidence") or []:
        if quote and quote not in user_text:
            evidence_valid = False
            warnings.append("evidence_not_literal")
    return evidence_valid, warnings


async def call_judge(client, args, messages, mode):
    kwargs = {
        "model": args.model,
        "messages": messages,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "seed": args.seed,
    }
    if mode in RESPONSE_FORMATS:
        kwargs["response_format"] = RESPONSE_FORMATS[mode]
    if args.extra_body:
        kwargs["extra_body"] = args.extra_body
    return await client.chat.completions.create(**kwargs)


async def request_annotation(client, args, messages):
    """Ask the judge until it returns a schema-valid object.

    Degrades the response format on refusal, retries a malformed reply, and
    returns `(parsed, completion, mode, error)`; `parsed` is None on failure.
    """
    last_error = None
    for mode in args.modes:
        for attempt in range(1, args.parse_retries + 1):
            try:
                completion = await call_judge(client, args, messages, mode)
            except BadRequestError as error:
                # The endpoint rejects this response_format; try a weaker one.
                last_error = f"{mode}: {error}"
                break
            except Exception as error:
                last_error = f"{mode}: {type(error).__name__}: {error}"
                await asyncio.sleep(min(2**attempt, 20))
                continue

            parsed = extract_json(completion.choices[0].message.content)
            problems = schema_problems(parsed) if parsed is not None else ["not_json"]
            if not problems:
                return parsed, completion, mode, None
            last_error = f"{mode}: {', '.join(problems)}"
    return None, None, None, last_error


async def annotate_one(client, sem, row, args):
    async with sem:
        key = str(row["key"])
        record = {
            "key": key,
            "group": row.get("group"),
            "sources": sources_of(row),
            "label": row.get("label"),
            "input_sha256": row.get("input_sha256"),
        }
        try:
            state = visible_state(row)
        except Exception as error:
            return {**record, "status": "invalid_input", "error": str(error)}

        user_text = json.dumps(state, ensure_ascii=False)
        if len(user_text) > args.max_input_chars:
            return {**record, "status": "too_long", "input_chars": len(user_text)}

        messages = [
            {"role": "system", "content": args.system},
            {"role": "user", "content": user_text},
        ]

        started = time.monotonic()
        parsed, completion, mode, last_error = await request_annotation(client, args, messages)
        latency = round((time.monotonic() - started) * 1000, 2)

        if parsed is not None:
            evidence_valid, warnings = validate(parsed, user_text)
            return {
                **record,
                "status": "ok",
                "parsed": parsed,
                "evidence_valid": evidence_valid,
                "schema_warnings": warnings,
                "raw": completion.choices[0].message.content,
                "model": args.model,
                "served_model": completion.model,
                "response_format": mode,
                "prompt_sha256": digest(args.system + "\n" + user_text),
                "input_sha256_prompt": digest(user_text),
                "latency_ms": latency,
                "usage": completion.usage.model_dump() if completion.usage else None,
            }

        return {**record, "status": "error", "error": last_error, "latency_ms": latency}


def load_pending(args):
    rows = []
    with args.input.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                rows.append(json.loads(line))
    if args.offset:
        rows = rows[args.offset :]
    if args.limit:
        rows = rows[: args.limit]

    done = set()
    kept = []
    total = 0
    if args.output.exists():
        with args.output.open("r", encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                total += 1
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get("status") in ("ok", "too_long"):
                    done.add(str(row["key"]))
                    kept.append(row)
        # Failed rows are retried on the next pass, so a transient error never
        # loses a candidate.
        if len(kept) != total:
            with args.output.open("w", encoding="utf-8") as sink:
                for row in kept:
                    sink.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return [row for row in rows if str(row["key"]) not in done]


async def run(args):
    pending = load_pending(args)

    if args.dry_run:
        print(
            json.dumps(
                {
                    "base_url": args.base_url,
                    "model": args.model,
                    "rows": len(pending),
                    "system_chars": len(args.system),
                    "response_formats": args.modes,
                }
            )
        )
        return

    sem = asyncio.Semaphore(args.concurrency)
    lock = asyncio.Lock()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    client = AsyncOpenAI(
        base_url=args.base_url,
        api_key=args.api_key or "EMPTY",
        timeout=args.timeout,
        max_retries=args.retries,
    )

    async def worker(row):
        result = await annotate_one(client, sem, row, args)
        async with lock:
            with args.output.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")
        return result

    counts = Counter()
    tokens = Counter()
    tasks = [worker(row) for row in pending]
    try:
        for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="annotate", unit="row"):
            result = await coro
            counts[result["status"]] += 1
            usage = result.get("usage") or {}
            for field in ("prompt_tokens", "completion_tokens"):
                tokens[field] += usage.get(field) or 0
    finally:
        await client.close()

    print(
        json.dumps(
            {
                "output": str(args.output),
                "processed": sum(counts.values()),
                "status": dict(counts),
                "tokens": dict(tokens),
            }
        )
    )


def main():
    load_dotenv(find_dotenv(usecwd=True))

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--system-file", type=Path, required=True)
    parser.add_argument("--base-url", default=env_value(BASE_URL_ENVS) or DEFAULT_BASE_URL)
    parser.add_argument("--api-key", default=env_value(API_KEY_ENVS))
    parser.add_argument("--json-mode", choices=["auto", "schema", "object", "none"], default="auto")
    parser.add_argument("--extra-body", type=json.loads, default=None, help="JSON merged into every request, e.g. vLLM chat_template_kwargs")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--max-input-chars", type=int, default=120000)
    parser.add_argument("--max-tokens", type=int, default=768)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20261109)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--retries", type=int, default=3, help="transport retries inside the client")
    parser.add_argument("--parse-retries", type=int, default=2, help="retries when the reply is not valid JSON")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    args.modes = ["schema", "object", "none"] if args.json_mode == "auto" else [args.json_mode]
    args.system = args.system_file.read_text(encoding="utf-8")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
