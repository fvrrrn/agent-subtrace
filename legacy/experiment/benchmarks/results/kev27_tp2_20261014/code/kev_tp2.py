"""TP=2 loader around the pinned Kev scoring and HTTP serving code; no weight edits."""

import argparse
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import time
import traceback

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AutoModel, DistributedConfig

from kev.api import SystemOneRequest, to_record, to_answers
from kev.checkpoint import Checkpoint
from kev.model import DecisionModel, PointerHead, is_hybrid, load_tokenizer, pad_id, rows_of
from kev.predictors import kernel_environment
from kev.serve import Server
from kev_hint_placement import LINK, RELATIONS, SAFETY, requests_for
from rjudge_kev_baseline import MODELS, SOURCE_COMMIT, sha


class TPDecisionModel(DecisionModel):
    def __init__(self, checkpoint, tokenizer, output):
        # The original constructor ends with .to(one_device), so use its scoring
        # methods but load the backbone directly with Transformers' official TP plan.
        torch.nn.Module.__init__(self)
        self.rank = dist.get_rank()
        self.device = f"cuda:{int(os.environ['LOCAL_RANK'])}"
        self.output = output
        self.lm = AutoModel.from_pretrained(checkpoint.path, dtype=torch.bfloat16,
            attn_implementation="sdpa", distributed_config=DistributedConfig(tp_size=2))
        self.pad_id = pad_id(tokenizer)
        self.hybrid = is_hybrid(self.lm.config)
        self.option_isolation = checkpoint.meta.option_isolation
        if not self.hybrid or self.option_isolation:
            raise ValueError("Expected the released hybrid Kev-27B without option isolation")
        self.head = PointerHead(self.lm.config.hidden_size, checkpoint.meta.head_dim).to(self.device)
        self.head.load_state_dict(checkpoint.meta.head)
        self.head.temperature = checkpoint.meta.temperature
        self.eval()
        self.worker_cache = {}
        self.readouts = []
        self.call_index = 0
        self.head.register_forward_hook(
            lambda module, inputs, output: self.readouts.append(output.detach().float().cpu().tolist()))

    def execute(self, encs, prefixes, keep):
        torch.cuda.set_device(self.device)
        self.readouts.clear()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode(), sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
            probabilities, made = super().probs_batch(encs, prefixes, keep)
        torch.cuda.synchronize()
        record = {"rank": self.rank, "seconds": time.perf_counter() - started,
                  "allocated_bytes": torch.cuda.memory_allocated(),
                  "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                  "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                  "probabilities": [[q.tolist() for q in p] for p in probabilities],
                  "temperature_scaled_logits": list(self.readouts)}
        observations = [None] * 2
        dist.all_gather_object(observations, record)
        if self.rank == 0:
            with (self.output / "native_calls.jsonl").open("a") as handle:
                handle.write(json.dumps({"call": self.call_index, "encodings": encs,
                    "cached": [p is not None for p in prefixes], "keep": keep,
                    "rank_observations": observations}) + "\n")
            self.call_index += 1
        return probabilities, made

    def probs_batch(self, encs, prefixes, keep):
        if self.rank != 0:
            raise RuntimeError("Only rank zero accepts serving requests")
        torch.cuda.set_device(self.device)
        dist.broadcast_object_list([{"encs": encs, "cached": [p is not None for p in prefixes],
                                     "keep": keep}], src=0, device=torch.device(self.device))
        return self.execute(encs, prefixes, keep)

    def worker_loop(self):
        while True:
            payload = [None]
            dist.broadcast_object_list(payload, src=0, device=torch.device(self.device))
            request = payload[0]
            if request is None:
                return
            encs, keep = request["encs"], request["keep"]
            keys = [tuple(rows_of(enc)[0]) for enc in encs]
            cached = [self.worker_cache[key] if hit else None
                      for key, hit in zip(keys, request["cached"])]
            self.worker_cache = {key: self.worker_cache[key] for key, hit, save in
                                 zip(keys, request["cached"], keep) if hit and save}
            probabilities, made = self.execute(encs, cached, keep)
            self.worker_cache = {key: prefix for key, prefix, save in zip(keys, made, keep) if save}
            del probabilities, made, cached


def preflight(server, output):
    rows = []
    graph = {"proposed_action": "E2", "highlighted_ancestors": ["E1"],
             "direct_parents": ["E1"],
             "edges": [{"source": "E1", "target": "E2", "relation": "USES_RESULT"}]}
    for repetitions in (32, 8192, 32768, 64000):
        state = ("Public reference text. " + "public " * repetitions +
                 "\nObserved E1: a public brochure was read. The user permits saving it locally. "
                 "Proposed E2: save that public brochure in a local file.")
        requests = {"link": {"model": "kev-latest", "state": state, "questions": {
            "link_E1_E2": {"type": "choice", "instructions": LINK.format(source="E1", target="E2"),
                            "criteria": RELATIONS}}}, **requests_for(state, graph)}
        for stage, request in requests.items():
            rec, mapping = to_record(SystemOneRequest.model_validate(request))
            probabilities, stats = server.probs(rec)
            row = {"stage": stage, "repetitions": repetitions, "request": request,
                   "response": {"answers": to_answers(probabilities, mapping), "stats": stats}}
            rows.append(row)
            with (output / "preflight.jsonl").open("a") as handle:
                handle.write(json.dumps(row) + "\n")
            print(json.dumps({"stage": stage, "repetitions": repetitions, "stats": stats}), flush=True)
    return [{"stage": r["stage"], "repetitions": r["repetitions"],
             "stats": r["response"]["stats"]} for r in rows]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8040)
    parser.add_argument("--serve", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl", timeout=timedelta(minutes=5))
    if dist.get_world_size() != 2:
        raise ValueError("This launcher requires exactly two TP ranks")
    spec = MODELS["27b"]
    summary = {"status": "loading", "created_utc": datetime.now(timezone.utc).isoformat(),
               "rank": rank, "model": spec, "source_commit": SOURCE_COMMIT,
               "settings": {"tp": 2, "dp": 1, "dtype": "bf16", "fused": False, "cuda_graphs": False,
                            "truncation": False, "prefix_cache": 1, "prefix_max_tokens": 65536,
                            "state_limit": 65536, "row_limit": 73728},
               "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
    def save():
        (args.output / f"rank_{rank}.json").write_text(json.dumps(summary, indent=2) + "\n")
    save()
    server = None
    try:
        torch.manual_seed(0)
        source = Path(__import__("kev").__file__).parent.parent
        if (source / "SOURCE_COMMIT").read_text().strip() != SOURCE_COMMIT:
            raise ValueError("Kev source mismatch")
        checkpoint = Checkpoint(f"{spec['repo']}@{spec['revision']}")
        valid = [None]
        if rank == 0:
            valid[0] = (checkpoint.full and checkpoint.saved_dtype() == "bfloat16"
                and checkpoint.weights_sha256() == spec["weights_sha"]
                and sha(checkpoint.file("head.pt")) == spec["head_sha"])
        dist.broadcast_object_list(valid, src=0)
        if not valid[0]:
            raise ValueError("Checkpoint hashes or saved dtype mismatch")
        tokenizer = load_tokenizer(checkpoint.meta.base, revision=checkpoint.meta.base_revision)
        started = time.perf_counter()
        model = TPDecisionModel(checkpoint, tokenizer, args.output)
        sharded = {name: {"global_shape": list(p.shape), "local_shape": list(p.to_local().shape),
                          "placements": [str(v) for v in p.placements]}
                   for name, p in model.lm.named_parameters() if isinstance(p, DTensor)}
        if not sharded:
            raise ValueError("The requested TP model has no DTensor-sharded parameters")
        summary.update(status="loaded", load_seconds=time.perf_counter() - started,
                       allocated_bytes=torch.cuda.memory_allocated(),
                       temperature=model.head.temperature, tp_plan=model.lm._tp_plan,
                       sharded_parameters=sharded, environment=kernel_environment(model, model.device))
        save()
        if rank:
            model.worker_loop()
        else:
            server = Server(checkpoint, tokenizer, model, model.device, release_date="2026-09-30",
                            truncate_states=False)
            summary["preflight"] = preflight(server, args.output)
            summary["status"] = "serving" if args.serve else "complete"
            save()
            if args.serve:
                import uvicorn
                from kev.serve import app
                app.state.server = server
                uvicorn.run(app, host="127.0.0.1", port=args.port)
            server.close()
            dist.broadcast_object_list([None], src=0)
        summary["status"] = "complete"
    except BaseException:
        summary.update(status="failed", error=traceback.format_exc())
        raise
    finally:
        save()
        if server is not None:
            server.close()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
