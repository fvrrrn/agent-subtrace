"""Frozen h0 replay with local choice-classifier adapters (no held-out access)."""
import argparse
import hashlib
import json
import platform
import sys
import time
import traceback
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace

import torch
from huggingface_hub import hf_hub_download
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parent
EXPERIMENT = ROOT.parent
sys.path.insert(0, str(EXPERIMENT / "evolution"))
sys.path.insert(0, str(EXPERIMENT))
from evaluate import metrics, run_fixture

MODELS = {
    "base": ("MoritzLaurer/ModernBERT-base-zeroshot-v2.0", "d421c4545a438fd006fb43f8b981c5d908faa1e1"),
    "large": ("MoritzLaurer/ModernBERT-large-zeroshot-v2.0", "a51e07b524299e309dd2b88d48b0cfa2bd9ec598"),
    "gliclass": ("knowledgator/gliclass-modern-base-v3.0", "ac369222ca4375ca66ebaf7fb5220f223514c035"),
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Adapter:
    def __init__(self, name, device, directory):
        self.name = name
        self.model_id, self.revision = MODELS[name]
        self.device = torch.device(device)
        self.directory = directory
        self.calls = 0
        self.cpu_fallback_count = 0
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, revision=self.revision)
        if name == "gliclass":
            from gliclass import GLiClassModel, ZeroShotClassificationPipeline
            from gliclass.config import GLiClassModelConfig
            config = GLiClassModelConfig.from_pretrained(self.model_id, revision=self.revision)
            config.encoder_config.reference_compile = False
            config.encoder_config._attn_implementation = "sdpa"
            self.model = GLiClassModel.from_pretrained(
                self.model_id, revision=self.revision, config=config, torch_dtype=torch.float32,
            ).to(self.device).eval()
            self.pipeline = ZeroShotClassificationPipeline(
                self.model, self.tokenizer, max_length=8192, classification_type="multi-label",
                device=self.device, progress_bar=False,
            )
            self.limit = config.encoder_config.max_position_embeddings
        else:
            config = AutoConfig.from_pretrained(self.model_id, revision=self.revision)
            config.reference_compile = False
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_id, revision=self.revision, config=config,
                attn_implementation="sdpa", torch_dtype=torch.float32,
            ).to(self.device).eval()
            self.limit = config.max_position_embeddings
            self.entailment_id = config.label2id["entailment"]
        if self.limit != 8192:
            raise ValueError(f"Unexpected encoder context: {self.limit}")

    def predict(self, state, questions, max_len=8192, head_max_len=8192):
        start = time.perf_counter()
        raw = {"call": self.calls, "state": state, "questions": questions,
               "started_utc": datetime.now(timezone.utc).isoformat(), "device": str(self.device)}
        self.calls += 1
        answers, details, lengths = {}, {}, []
        try:
            with torch.inference_mode():
                for key, question in questions.items():
                    if question["type"] != "choice":
                        raise ValueError("Only choice questions are supported")
                    identities = list(question["criteria"])
                    labels = [f"{identity}: {question['criteria'][identity]}" for identity in identities]
                    premise = f"{state}\n\nQuestion: {question['instructions']}"
                    if self.name == "gliclass":
                        encoded_text = self.pipeline.pipe.prepare_input(premise, labels)
                        ids = self.tokenizer(encoded_text, truncation=False)["input_ids"]
                        lengths.append(len(ids))
                        if len(ids) > min(max_len, self.limit):
                            raise ValueError(f"GLiClass context overflow: {len(ids)}")
                        # Native pipeline uses truncation=True; equality checks ensure it removed nothing.
                        prepared = self.pipeline.pipe.prepare_inputs([premise], labels, True)
                        actual = prepared["input_ids"][0].cpu().tolist()
                        if actual != ids:
                            raise ValueError("GLiClass native tokenization differs or truncates")
                        captured = {}
                        def capture(module, inputs, output):
                            captured["logits"] = output.logits.float().cpu().tolist()
                        hook = self.model.register_forward_hook(capture)
                        try:
                            native = self.pipeline(premise, labels, threshold=0.0)[0]
                        finally:
                            hook.remove()
                        by_label = {item["label"]: item["score"] for item in native}
                        scores = torch.tensor([by_label[label] for label in labels], dtype=torch.float64)
                        probabilities = scores / scores.sum()
                        details[key] = {"premise": premise, "labels": labels, "encoded_text": encoded_text,
                                        "input_ids": ids, "native": native, **captured,
                                        "scoring": "sigmoid(logit) / sum(sigmoid(logits)); native independent scores retained"}
                    else:
                        encoded = self.tokenizer([premise] * len(labels), labels, padding=True,
                                                 truncation=False, return_tensors="pt")
                        pair_lengths = encoded["attention_mask"].sum(-1).tolist()
                        lengths.extend(pair_lengths)
                        if max(pair_lengths) > min(max_len, self.limit):
                            raise ValueError(f"NLI context overflow: {pair_lengths}")
                        logits = self.model(**encoded.to(self.device)).logits.float().cpu()
                        probabilities = logits[:, self.entailment_id].softmax(0)
                        details[key] = {"premise": premise, "hypotheses": labels,
                                        "input_ids": encoded["input_ids"].cpu().tolist(),
                                        "pair_tokens": pair_lengths, "logits": logits.tolist(),
                                        "independent_entailment": logits.softmax(-1)[:, self.entailment_id].tolist(),
                                        "scoring": "softmax across candidate entailment logits (not independent NLI probabilities)"}
                    values = dict(zip(identities, probabilities.tolist()))
                    if not torch.isfinite(probabilities).all():
                        raise ValueError("Nonfinite probabilities")
                    answers[key] = {"type": "choice", "choice": max(values, key=values.get), "probabilities": values}
            if self.device.type == "mps":
                torch.mps.synchronize()
            output = {"answers": answers, "usage": {"input_tokens": sum(lengths),
                      "sequence_tokens": lengths, "max_sequence_tokens": max(lengths),
                      "encoder_limit": self.limit, "truncated": False, "output_tokens": 0},
                      "adapter": details}
            raw["output"] = output
            return output
        except Exception:
            raw["error"] = traceback.format_exc()
            raw["partial_adapter"] = details
            raise
        finally:
            raw["seconds"] = time.perf_counter() - start
            with (self.directory / "calls.jsonl").open("a") as handle:
                handle.write(json.dumps(raw, ensure_ascii=False) + "\n")


def edge_metrics(fixtures):
    tp = sum(f["edges"]["tp"] for f in fixtures)
    predicted = sum(f["edges"]["predicted"] for f in fixtures)
    gold = sum(f["edges"]["gold"] for f in fixtures)
    return {"tp": tp, "fp": predicted - tp, "fn": gold - tp,
            "precision": tp / predicted if predicted else 0,
            "recall": tp / gold if gold else 0,
            "f1": 2 * tp / (predicted + gold) if predicted + gold else 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=MODELS)
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    directory = ROOT / "results" / f"{stamp}_{args.model}_{args.device}"
    directory.mkdir(parents=True)
    report = {"model": MODELS[args.model][0], "revision": MODELS[args.model][1],
              "device": args.device, "started_utc": datetime.now(timezone.utc).isoformat(),
              "environment": {"python": platform.python_version(), "platform": platform.platform(),
                              **{p: version(p) for p in ("laya", "torch", "transformers", "gliclass", "huggingface-hub")}},
              "settings": {"max_len": 8192, "head_max_len": 8192, "link_threshold": .2,
                           "violation_threshold": .5, "dtype": "float32", "reference_compile": False,
                           "attention": "sdpa", "seed": 0},
              "timing_caveat": "Parent Laya evaluations may concurrently use this MPS device; timings are not isolated benchmarks.",
              "sha256": {}, "groups": {}}
    for source in (Path(__file__), ROOT / "pyproject.toml", ROOT / "uv.lock",
                   EXPERIMENT / "run.py", EXPERIMENT / "evolution/evaluate.py",
                   EXPERIMENT / "evolution/harness.py", EXPERIMENT / "evolution/candidates/h0.json",
                   EXPERIMENT / "trajectory.json", EXPERIMENT / "evolution/dev.json"):
        report["sha256"][str(source.relative_to(EXPERIMENT))] = digest(source)
        (directory / source.name).write_bytes(source.read_bytes())
    start = time.perf_counter()
    try:
        for filename in ("README.md", "config.json", "tokenizer_config.json"):
            path = hf_hub_download(report["model"], filename, revision=report["revision"])
            (directory / ("model_card.md" if filename == "README.md" else filename)).write_bytes(Path(path).read_bytes())
        torch.manual_seed(0)
        agent = Adapter(args.model, args.device, directory)
        report["load_seconds"] = time.perf_counter() - start
        candidate = json.loads((EXPERIMENT / "evolution/candidates/h0.json").read_text())
        groups = {"original": [json.loads((EXPERIMENT / "trajectory.json").read_text())],
                  "dev": json.loads((EXPERIMENT / "evolution/dev.json").read_text())["fixtures"]}
        inference_start = time.perf_counter()
        for name, fixtures in groups.items():
            measured = [run_fixture(agent, fixture, candidate, SimpleNamespace(max_len=8192, head_max_len=8192))
                        for fixture in fixtures]
            rows = [row for result in measured for row in result["records"] if row["evaluated"]]
            pairs = {}
            for fixture in measured:
                if fixture["pair"]:
                    pairs.setdefault(fixture["pair"], []).append(fixture)
            report["groups"][name] = {"fixtures": measured, "graph": metrics(rows, "graph"),
                                      "full": metrics(rows, "full"), "edges": edge_metrics(measured),
                                      "pairs": len(pairs), "paired_all_correct": sum(
                                          all(f["graph"]["fp"] + f["graph"]["fn"] == 0 for f in pair)
                                          for pair in pairs.values())}
        report["inference_seconds"] = time.perf_counter() - inference_start
        report["calls"] = agent.calls
        report["status"] = "complete"
        report["cpu_fallback_count"] = agent.cpu_fallback_count
    except Exception:
        report["status"] = "failed"
        report["error"] = traceback.format_exc()
    finally:
        report["total_seconds"] = time.perf_counter() - start
        report["ended_utc"] = datetime.now(timezone.utc).isoformat()
        (directory / "trace.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        summary = {key: value for key, value in report.items() if key not in ("groups", "sha256")}
        summary["groups"] = {key: {k: v for k, v in value.items() if k != "fixtures"}
                             for key, value in report["groups"].items()}
        (directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps({"directory": str(directory), **summary}), flush=True)


if __name__ == "__main__":
    main()
