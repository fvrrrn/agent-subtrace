"""Pack the jsonl splits into a Hugging Face `DatasetDict`.

The visible state is free-form (`tool_interfaces`, `observed_history` and
`proposed_action` hold arbitrary nested JSON), which pyarrow cannot unify across
rows. Those three fields are therefore stored as JSON strings, `user_request` as
plain text. `json.dumps(value, ensure_ascii=False)` on the jsonl field restores
the exact string; `data/*.jsonl` stays the canonical structured form.

    uv run --with datasets python3 make_hf_dataset.py --data-dir dataset_v1
"""
import argparse
import json
from pathlib import Path

from datasets import Dataset, DatasetDict, Features, Sequence, Value

SPLITS = ("train", "dev", "test", "ood")

FEATURES = Features(
    {
        "key": Value("string"),
        "group": Value("string"),
        "split": Value("string"),
        "slice": Value("string"),
        "input": {
            # Alphabetical order matches the canonical `state` serialization, so
            # json.dumps of the rebuilt dict reproduces `input_sha256` exactly.
            "observed_history": Value("string"),
            "proposed_action": Value("string"),
            "tool_interfaces": Value("string"),
            "user_request": Value("string"),
        },
        "label": Value("int8"),
        "uncertain": Value("bool"),
        "facets": Sequence(Value("string")),
        "severity": Value("string"),
        "reversibility": Value("string"),
        "actor": Value("string"),
        "authorization": Value("string"),
        "evidence": Value("string"),
        "role": Value("string"),
        "sources": Sequence(Value("string")),
        "input_sha256": Value("string"),
        "annotation": {
            "model": Value("string"),
            "prompt": Value("string"),
            "prompt_sha256": Value("string"),
            "confidence": Value("float32"),
            "reason": Value("string"),
            "status": Value("string"),
        },
    }
)

JSON_INPUT_FIELDS = ("tool_interfaces", "observed_history", "proposed_action")


def convert(row):
    state = row["input"]
    row = dict(row)
    row["input"] = {
        name: (value if name == "user_request" else json.dumps(value, ensure_ascii=False))
        for name, value in ((name, state.get(name)) for name in FEATURES["input"])
    }
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()
    out_dir = args.out_dir or args.data_dir / "hf"

    splits = {}
    for split in SPLITS:
        path = args.data_dir / "data" / f"{split}.jsonl"
        rows = [convert(json.loads(line)) for line in path.open("r", encoding="utf-8") if line.strip()]
        splits[split] = Dataset.from_list(rows, features=FEATURES)
        print(f"{split}: {len(splits[split])} rows")

    dataset = DatasetDict(splits)
    dataset.save_to_disk(str(out_dir))

    summary = {
        split: {
            "rows": len(dataset[split]),
            "unsafe": sum(1 for value in dataset[split]["label"] if value == 1),
            "safe": sum(1 for value in dataset[split]["label"] if value == 0),
            "uncertain": sum(1 for value in dataset[split]["label"] if value is None),
        }
        for split in SPLITS
    }
    print(json.dumps(summary, indent=2))
    print(f"saved to {out_dir}")


if __name__ == "__main__":
    main()
