r"""Add a judge label column to a parquet dataset.

Reads a parquet file (or a directory of them), asks the judge for every row that
has no value in `<model>_judge_label` yet, and writes the same rows to the output
with that column filled. All other columns are copied untouched, so labels from
different judges accumulate side by side and are compared column to column.

The column holds the verdict: `safe`, `unsafe` or `uncertain`. A row the judge
failed on stays null and is retried on the next run.

    uv run python scripts/judges/annotate.py \
        --input dataset --output judged \
        --model qwen/qwen3-235b-a22b-2507 \
        --system-file scripts/dataset/flash_next/prompt.txt

Credentials come from the environment, with `.env` loaded from the working
directory upwards: `JUDGE_API_KEY` / `OPENAI_API_KEY` / `OPENROUTER_API_KEY`,
and `JUDGE_BASE_URL` / `OPENAI_BASE_URL` / `OPENROUTER_BASE_URL`.

Structured output degrades on demand: `json_schema` first, then `json_object`,
then plain text with JSON extraction. `--json-mode` pins one of them.
"""
import argparse
import asyncio
import hashlib
import json
import os
import re
from collections import Counter
from pathlib import Path

from datasets import Dataset, Value
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


def label_column_for(model):
    slug = re.sub(r"[^a-z0-9]+", "_", model.lower()).strip("_")
    return f"{slug}_judge_label"


def env_value(names):
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return None


def flatten_usage(usage, prefix=""):
    """Flatten a usage object into dotted counters, e.g. `cost`, `prompt_tokens`."""
    flat = {}
    for key, value in (usage or {}).items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(flatten_usage(value, f"{name}."))
        elif isinstance(value, (int, float)) and not isinstance(value, bool) and value:
            flat[name] = value
    return flat


def parquet_files(path):
    return sorted(path.glob("*.parquet")) if path.is_dir() else [path]


def visible_state(row_input):
    state = {}
    for name, value in row_input.items():
        state[name] = value if name == "user_request" else json.loads(value)
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
                last_error = f"{type(error).__name__}: {error}"
                await asyncio.sleep(min(2**attempt, 20))
                continue

            parsed = extract_json(completion.choices[0].message.content)
            problems = schema_problems(parsed) if parsed is not None else ["not_json"]
            if not problems:
                return parsed, completion, mode, None
            last_error = f"{mode}: {', '.join(problems)}"
    return None, None, None, last_error


async def annotate_rows(client, args, states):
    """Return one verdict per state (None where the judge failed)."""
    sem = asyncio.Semaphore(args.concurrency)
    verdicts = [None] * len(states)
    errors = Counter()
    usage = Counter()
    samples = []
    quoted = 0

    async def worker(index, state):
        async with sem:
            user_text = json.dumps(state, ensure_ascii=False)
            if len(user_text) > args.max_input_chars:
                return index, None, user_text, "too_long", None
            messages = [
                {"role": "system", "content": args.system},
                {"role": "user", "content": user_text},
            ]
            parsed, completion, mode, error = await request_annotation(client, args, messages)
            usage = completion.usage.model_dump() if completion and completion.usage else None
            return index, parsed, user_text, error, usage

    tasks = [worker(index, state) for index, state in enumerate(states)]
    bar = tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="annotate", unit="row")
    for coro in bar:
        index, parsed, user_text, error, response_usage = await coro
        if parsed is None:
            errors[error.split(":")[0]] += 1
            if error not in samples and len(samples) < 3:
                samples.append(error)
            continue
        verdicts[index] = parsed["verdict"]
        for name, value in flatten_usage(response_usage).items():
            usage[name] += value
        if usage["cost"]:
            bar.set_postfix_str(f"${usage['cost']:.4f}")
        if any(quote and quote not in user_text for quote in parsed.get("evidence") or []):
            quoted += 1

    return verdicts, errors, quoted, usage, samples


async def process_file(client, args, source, target):
    dataset = Dataset.from_parquet(str(source))
    rows = len(dataset)
    keys = dataset["key"]
    inputs = dataset["input"]

    labels = list(dataset[args.label_column]) if args.label_column in dataset.column_names else [None] * rows
    if target.exists():
        previous = Dataset.from_parquet(str(target))
        if args.label_column in previous.column_names:
            known = dict(zip(previous["key"], previous[args.label_column]))
            labels = [label or known.get(key) for key, label in zip(keys, labels)]

    pending = [index for index, label in enumerate(labels) if label is None]
    if args.limit:
        pending = pending[: args.limit]

    summary = {
        "file": str(source),
        "output": str(target),
        "column": args.label_column,
        "rows": rows,
        "pending": len(pending),
    }
    if args.dry_run:
        return summary

    errors = Counter()
    states, indices = [], []
    for index in pending:
        try:
            states.append(visible_state(inputs[index]))
        except Exception as error:
            errors[type(error).__name__] += 1
            continue
        indices.append(index)

    verdicts, judge_errors, quoted, usage, samples = await annotate_rows(client, args, states)
    for index, verdict in zip(indices, verdicts):
        labels[index] = verdict

    if args.label_column in dataset.column_names:
        dataset = dataset.remove_columns([args.label_column])
    dataset = dataset.add_column(args.label_column, labels)
    dataset = dataset.cast_column(args.label_column, Value("string"))
    target.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(str(target))

    errors.update(judge_errors)
    summary.update(
        {
            "labeled": sum(label is not None for label in labels),
            "errors": dict(errors),
            "error_samples": samples,
            "evidence_not_literal": quoted,
            "usage": dict(usage),
            "prompt_sha256": hashlib.sha256(args.system.encode()).hexdigest(),
        }
    )
    return summary


async def run(args):
    files = parquet_files(args.input)
    if args.input.is_dir() and args.output.exists() and not args.output.is_dir():
        raise SystemExit(f"--output must be a directory when --input is one: {args.output}")

    client = None if args.dry_run else AsyncOpenAI(
        base_url=args.base_url,
        api_key=args.api_key or "EMPTY",
        timeout=args.timeout,
        max_retries=args.retries,
    )
    try:
        for source in files:
            target = args.output / source.name if args.input.is_dir() else args.output
            summary = await process_file(client, args, source, target)
            print(json.dumps(summary, ensure_ascii=False))
    finally:
        if client is not None:
            await client.close()


def main():
    load_dotenv(find_dotenv(usecwd=True))

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, required=True, help="parquet file or directory of them")
    parser.add_argument("--output", type=Path, required=True, help="same kind as --input")
    parser.add_argument("--model", required=True)
    parser.add_argument("--system-file", type=Path, required=True)
    parser.add_argument("--label-column", default=None, help="defaults to <model>_judge_label")
    parser.add_argument("--base-url", default=env_value(BASE_URL_ENVS) or DEFAULT_BASE_URL)
    parser.add_argument("--api-key", default=env_value(API_KEY_ENVS))
    parser.add_argument("--json-mode", choices=["auto", "schema", "object", "none"], default="auto")
    parser.add_argument("--extra-body", type=json.loads, default=None, help="JSON merged into every request, e.g. vLLM chat_template_kwargs")
    parser.add_argument("--limit", type=int, default=0, help="label at most N pending rows per file")
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
    args.label_column = args.label_column or label_column_for(args.model)
    args.system = args.system_file.read_text(encoding="utf-8")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
