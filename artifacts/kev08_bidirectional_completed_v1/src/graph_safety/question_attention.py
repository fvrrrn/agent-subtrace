"""Bias SAFETY branch queries toward a graph-selected set of past events."""

import argparse
import hashlib
from pathlib import Path
import random

from . import attention_pilot as core
from .io import read_json, sha256, write_json

ARMS = ("baseline", "branch_last_03", "branch_all_03", "readout_last_03",
        "readout_all_03", "branch_last_10", "random_branch_last_03")


class QuestionBias(core.GraphBias):
    def set_state(self, request, graph):
        from kev.model import rows_of
        super().set_state(request, graph)
        _, _, encoding, _ = self.backend.validate(request)
        _, _, branches = rows_of(encoding)
        if len(branches) != 1:
            raise ValueError("Exactly one isolated SAFETY question required")
        self.branch_positions = branches[0]["pos"]
        self.readout_positions = {pos for pos, segment, option in zip(encoding["pos"], encoding["seg"], encoding["opt"])
                                  if segment == 1 and (option >= 0 or option == -2)}
        target = list(graph)[-1]
        ancestors, todo = set(), list(graph[target])
        while todo:
            event = todo.pop()
            if event not in ancestors:
                ancestors.add(event)
                todo.extend(graph[event])
        self.selected = [e for e in graph if e in ancestors]
        past = [e for e in graph if e != target]
        seed = int(hashlib.sha256(request["state"].encode()).hexdigest()[:16], 16)
        self.random_selected = sorted(random.Random(seed).sample(past, len(self.selected)), key=past.index)
        def tokens(events):
            return sum(b - a for event in events for a, b in self.spans[event])
        self.selection_audit = {"target": target, "ancestors": self.selected,
                                "random_events": self.random_selected,
                                "overlap_events": sorted(set(self.selected) & set(self.random_selected)),
                                "ancestor_tokens": tokens(self.selected), "random_tokens": tokens(self.random_selected),
                                "branch_positions": self.branch_positions,
                                "readout_positions": sorted(self.readout_positions)}

    def configure(self, arm, alpha=.3):
        # The generic runner's zero gate installs all hooks but performs no bias.
        if arm == "all_attention" and alpha == 0:
            super().configure(arm, alpha)
            return
        self.arm = arm
        mode = "all_attention" if "_all_" in arm else "last_attention"
        super().configure("baseline" if arm == "baseline" else mode, 1.0 if arm.endswith("_10") else alpha)

    def hook(self, index):
        def before(module, args, kwargs):
            import torch
            if index not in self.active or self.alpha == 0:
                return
            hidden = kwargs.get("hidden_states", args[0] if args else None)
            positions = kwargs.get("position_ids")
            if positions is None or hidden is None:
                raise ValueError("Missing attention inputs")
            flat = positions.reshape(-1, positions.shape[-1])
            if int(flat.max().item()) < self.length:
                return  # State is encoded identically for every arm.
            if hidden.shape[0] != 1 or flat[0].tolist() != self.branch_positions:
                raise ValueError("Unsupported branch positions/batching")
            self.counts[str(index)] = self.counts.get(str(index), 0) + 1
            selected = self.random_selected if self.arm.startswith("random_") else self.selected
            if not selected:
                return  # Empty ancestor set is exactly the native computation.
            query_indices = [i for i, pos in enumerate(self.branch_positions)
                             if not self.arm.startswith("readout_") or pos in self.readout_positions]
            if not query_indices:
                raise ValueError("No selected question queries")
            query_length = hidden.shape[1]
            key_length = self.length + query_length
            original = kwargs.get("attention_mask")
            if original is None:
                mask = torch.zeros((1, 1, query_length, key_length), device=hidden.device, dtype=hidden.dtype)
                allowed = torch.arange(key_length, device=hidden.device)[None, :] <= flat[0, :, None]
                mask.masked_fill_(~allowed, float("-inf"))
            elif original.dtype == torch.bool:
                mask = torch.zeros_like(original, dtype=hidden.dtype).masked_fill_(~original, float("-inf"))
            else:
                mask = original.clone()
            if mask.shape[-2:] != (query_length, key_length):
                raise ValueError("Unexpected cached branch mask")
            for event in selected:
                for start, end in self.spans[event]:
                    mask[..., query_indices, start:end] += self.alpha
            kwargs["attention_mask"] = mask
            return args, kwargs
        return before


def prepare(root, source):
    core.prepare(root, source)
    protocol = read_json(root / "protocol.json")
    protocol.update(arms=ARMS,
        intervention="Only SAFETY branch queries attend more strongly to tokens of all ancestors of current action. No edge-dependent weights, no state prefill modification, no extra graph text. Current action/user/profile not boosted. Empty ancestor sets are native no-ops.",
        variants={"branch_last_03": "All branch tokens, last ordinary attention, +0.3",
                  "branch_all_03": "All branch tokens, all ordinary attention, +0.3",
                  "readout_last_03": "All option-content tokens and decide token, last ordinary attention, +0.3",
                  "readout_all_03": "All option-content tokens and decide token, all ordinary attention, +0.3",
                  "branch_last_10": "All branch tokens, last ordinary attention, +1.0",
                  "random_branch_last_03": "All branch tokens, last ordinary attention, +0.3 to deterministic random past events, matched event count"},
        evaluation="Same metrics as preceding pilot, Holm over six AP-vs-baseline contrasts. Additionally paired real-vs-random AP comparison descriptive. No adaptive extension of strengths or variants.",
        limitations=["Exploratory previously studied 40-case cohort.",
                     "Random control matches event count, not token count; overlaps retained and audited, including full overlap when all history is selected.",
                     "Explicit additive mask may alter SDPA kernel dispatch; alpha=0 gate checks hook identity only.",
                     "Predicted graph fixed from completed NOUL run with threshold 0.50.",
                     "Best BA on same labels is diagnostic, not held-out calibration."])
    write_json(root / "protocol.json", protocol)


def aggregate(root):
    import numpy as np
    from sklearn.metrics import average_precision_score
    from .io import read_jsonl
    core.aggregate(root)
    result = read_json(root / "results.json")
    rows = [r for shard in (0, 1) for r in read_jsonl(root / f"shard_{shard}/records.jsonl")]
    y = np.array([r["label"] for r in rows])
    a = np.array([r["scores"]["branch_last_03"] for r in rows])
    b = np.array([r["scores"]["random_branch_last_03"] for r in rows])
    rng = np.random.default_rng(2026101904)
    delta = average_precision_score(y, a) - average_precision_score(y, b)
    groups = [np.flatnonzero(y == c) for c in (0, 1)]
    boot = []
    for _ in range(5000):
        idx = np.concatenate([rng.choice(g, len(g), replace=True) for g in groups])
        boot.append(average_precision_score(y[idx], a[idx]) - average_precision_score(y[idx], b[idx]))
    result["real_minus_random_AP_descriptive"] = {"delta": float(delta), "ci95": np.quantile(boot, [.025, .975]).tolist()}
    result["selection_audit"] = {"empty_ancestor_actions": sum(not s["selection"]["ancestors"] for r in rows for s in r["steps"]),
                                 "identical_random_sets": sum(set(s["selection"]["ancestors"]) == set(s["selection"]["random_events"]) for r in rows for s in r["steps"])}
    write_json(root / "results.json", result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "run", "aggregate"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--shard", type=int, choices=(0, 1))
    args = parser.parse_args()
    if args.phase == "prepare":
        prepare(args.root, args.source)
    elif args.phase == "run":
        core.run(args.root, args.shard, bias_class=QuestionBias)
    else:
        aggregate(args.root)


if __name__ == "__main__":
    main()
