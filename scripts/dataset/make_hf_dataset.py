"""Turn the staged jsonl rows into the released parquet files.

`build_dataset.py` writes one staging jsonl per split; this script converts them
into `<out-dir>/{train,dev,test,ood}.parquet` (Hugging Face `datasets` schema,
read back with `load_dataset("parquet", ...)`), records the file hashes in
`<out-dir>/manifest.json`, and removes the staging directory. The release has no
jsonl and no arrow copy.

The visible state is free-form (`tool_interfaces`, `observed_history` and
`proposed_action` hold arbitrary nested JSON), which pyarrow cannot unify across
rows. Those three fields are therefore stored as JSON strings and `user_request`
as plain text. The field order is alphabetical, so `json.dumps` of the rebuilt
dict reproduces `input_sha256` exactly.

    uv run --with datasets python3 make_hf_dataset.py \
        --staging-dir data/staging --out-dir dataset
"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

from datasets import Dataset, Features, Sequence, Value

SPLITS = ("train", "dev", "test", "ood")

def features(label_column):
    return Features(
        {
            "key": Value("string"),
            "group": Value("string"),
            "split": Value("string"),
            "slice": Value("string"),
            "input": {
                "observed_history": Value("string"),
                "proposed_action": Value("string"),
                "tool_interfaces": Value("string"),
                "user_request": Value("string"),
            },
            label_column: Value("string"),
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


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def convert(row, schema):
    state = row["input"]
    row = dict(row)
    row["input"] = {
        name: (value if name == "user_request" else json.dumps(value, ensure_ascii=False))
        for name, value in ((name, state.get(name)) for name in schema["input"])
    }
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staging-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    manifest_path = args.out_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    label_column = manifest["label_column"]
    schema = features(label_column)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows_per_split = {}
    for split in SPLITS:
        rows = [
            convert(json.loads(line), schema)
            for line in (args.staging_dir / f"{split}.jsonl").open("r", encoding="utf-8")
            if line.strip()
        ]
        dataset = Dataset.from_list(rows, features=schema)
        dataset.to_parquet(str(args.out_dir / f"{split}.parquet"))
        rows_per_split[split] = len(dataset)
        print(f"{split}: {len(dataset)} rows")

    manifest["parquet"] = {
        "format": "parquet, one file per split, read with datasets.load_dataset('parquet', ...)",
        "input_encoding": "observed_history, proposed_action, tool_interfaces as JSON strings; user_request as text",
        "builder": {"path": str(Path(__file__)), "sha256": sha256_file(Path(__file__))},
        "splits": {
            split: {
                "rows": rows_per_split[split],
                "sha256": sha256_file(args.out_dir / f"{split}.parquet"),
            }
            for split in SPLITS
        },
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    shutil.rmtree(args.staging_dir)
    print(json.dumps({"out_dir": str(args.out_dir), "rows": rows_per_split}))


if __name__ == "__main__":
    main()
