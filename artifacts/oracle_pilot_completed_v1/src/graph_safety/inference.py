"""Optional native Kev adapter; environment-specific dependencies load lazily.

The adapter pins the checkpoint and records every native request and response.
It does not load benchmark labels, choose prompts or change thresholds.
"""

from contextlib import AbstractContextManager
import json
from pathlib import Path
import time

from .io import sha256


class KevBackend(AbstractContextManager):
    def __init__(self, checkpoint, audit_path):
        self.checkpoint = checkpoint
        self.audit_path = Path(audit_path)
        self.server = None
        self.call_number = 0
        self.logits = []

    def __enter__(self):
        import torch
        import kev
        from kev.checkpoint import Checkpoint, LoadOptions
        from kev.predictors import kernel_environment
        from kev.serve import Server

        if self.audit_path.exists():
            raise FileExistsError(self.audit_path)
        spec = self.checkpoint
        source = Path(kev.__file__).parent.parent
        if (source / "SOURCE_COMMIT").read_text().strip() != spec["source_commit"]:
            raise ValueError("Kev source revision mismatch")
        checkpoint = Checkpoint(f"{spec['repo']}@{spec['revision']}")
        if checkpoint.weights_sha256() != spec["weights_sha"] or sha256(checkpoint.file("head.pt")) != spec["head_sha"]:
            raise ValueError("Checkpoint hash mismatch")
        if checkpoint.meta.base != spec["base"] or checkpoint.meta.base_revision != spec["base_revision"]:
            raise ValueError("Base revision mismatch")
        torch.manual_seed(0)
        tokenizer, model = checkpoint.load("cuda", LoadOptions(dtype=torch.bfloat16, merge=True, attn="sdpa", backend="torch", fused=True, cuda_graphs=True))
        self.server = Server(checkpoint, tokenizer, model, "cuda", truncate_states=False)
        self.environment = {"kernels": kernel_environment(model, "cuda"), "temperature": model.head.temperature}
        self.hook = model.head.register_forward_hook(lambda module, inputs, output: self.logits.append(output.detach().float().cpu().tolist()))
        original_many = model.head.many

        def capture_many(h_decide, h_opts, owner):
            logits = original_many(h_decide, h_opts, owner)
            cpu_logits, cpu_owner = logits.detach().float().cpu(), owner.cpu()
            self.logits.extend(cpu_logits[cpu_owner == i].tolist() for i in range(len(h_decide)))
            return logits

        model.head.many = capture_many
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        return self

    def validate(self, request):
        from kev.api import SystemOneRequest, to_record
        from kev.model import ContextOverflow, encode, rows_of

        record, mapping = to_record(SystemOneRequest.model_validate(request))
        encoding = encode(self.server.tok, record, max_state=65536, max_branch=73728, strict=True)
        state, _, branches = rows_of(encoding)
        if max(len(branch["ids"]) for branch in branches) > 8192:
            raise ContextOverflow("Question branch exceeds 8192 tokens")
        return record, mapping, encoding, {"state": len(state), "branches": [len(branch["ids"]) for branch in branches]}

    def predict(self, case_id, arm, request):
        from kev.api import to_answers

        audit = {"call": self.call_number, "case_id": case_id, "arm": arm, "request": request}
        self.call_number += 1
        try:
            record, mapping, encoding, lengths = self.validate(request)
            self.logits.clear()
            started = time.perf_counter()
            probabilities, stats = self.server.probs(record)
            seconds = time.perf_counter() - started
            self.server.wait_idle()
            if len(self.logits) != len(mapping):
                raise ValueError("Pointer capture count mismatch")
            audit.update(mapping=mapping, encoding=encoding, lengths=lengths, seconds=seconds, stats=stats,
                         native_probabilities=probabilities, temperature_scaled_logits=list(self.logits),
                         answers=to_answers(probabilities, mapping))
            return {k: audit[k] for k in ("call", "mapping", "native_probabilities", "lengths", "seconds", "answers")}
        except Exception as error:
            audit["error"] = {"type": type(error).__name__, "message": str(error)}
            raise
        finally:
            with self.audit_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(audit, ensure_ascii=False, allow_nan=False) + "\n")

    def __exit__(self, exc_type, exc, traceback):
        if self.server is not None:
            self.server.close()
            self.hook.remove()
            del self.server.model.head.many
        return False
