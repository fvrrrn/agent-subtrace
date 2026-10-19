"""Highlight action, dependency evidence and available user instructions together."""

import argparse
from pathlib import Path
import re

from . import attention_pilot as core
from . import question_attention as question
from .io import read_json, write_json

ARMS = ("baseline", "current_only", "current_and_ancestors", "current_ancestors_user")


class ContextBias(question.QuestionBias):
    def set_state(self, request, graph):
        from kev.model import _SPECIAL_RE
        super().set_state(request, graph)
        self.ancestors = list(self.selected)
        self.current = list(graph)[-1]
        text = _SPECIAL_RE.sub(r"<¦\1¦>", request["state"])
        encoded = self.backend.server.tok(text, add_special_tokens=False, return_offsets_mapping=True)
        self.user_ids = []
        for number, match in enumerate(re.finditer(r'<message role="user">.*?</message>', text, re.DOTALL)):
            indices = [i + 1 for i, (start, end) in enumerate(encoded.offset_mapping)
                       if end > start and start >= match.start() and end <= match.end()]
            if not indices or indices != list(range(indices[0], indices[-1] + 1)):
                raise ValueError("Invalid user instruction token span")
            event = f"USER_{number}"
            self.spans[event] = [(indices[0], indices[-1] + 1)]
            self.user_ids.append(event)
        self.selection_audit.update(current=self.current, user_messages=self.user_ids,
                                    selected_by_arm={"current_only": [self.current],
                                                     "current_and_ancestors": [self.current] + self.ancestors,
                                                     "current_ancestors_user": [self.current] + self.ancestors + self.user_ids})

    def configure(self, arm, alpha=.3):
        if arm == "all_attention" and alpha == 0:
            super().configure(arm, alpha)
            return
        if arm != "baseline":
            self.selected = self.selection_audit["selected_by_arm"][arm]
        super().configure("baseline" if arm == "baseline" else "branch_all_03", alpha)


def prepare(root, source):
    core.prepare(root, source)
    protocol = read_json(root / "protocol.json")
    protocol.update(arms=ARMS,
        intervention="SAFETY branch queries receive +0.3 attention bias to selected token spans in ALL ordinary attention blocks. State prefill unchanged. Same raw text for all arms, frozen source graphs. No event-to-event bias.",
        variants={"baseline": "No bias", "current_only": "Only current proposed action; focus control",
                  "current_and_ancestors": "Current proposed action and all graph ancestors, deduplicated",
                  "current_ancestors_user": "Current action, ancestors, and ALL visible chronological user messages; no heuristic extraction of permissions"},
        evaluation="AP/AUC/F1/BA at .50 and descriptive best BA; paired AP bootstrap and permutations versus baseline, Holm across three contrasts. Additional contrasts of each evidence arm versus current_only; separate exploratory Holm family.",
        limitations=["Previously studied 40 cases; hypotheses chosen after earlier null results, exploratory.",
                     "All visible user messages are highlighted, including requests that may themselves be harmful; highlighting is not authorization.",
                     "Equal per-token boost; arms select different token counts.",
                     "Fixed alpha .3, no adaptive search, frozen LINK .50 graph.",
                     "Numerical kernel changes from explicit attention mask remain possible."])
    write_json(root / "protocol.json", protocol)


def aggregate(root):
    import numpy as np
    from sklearn.metrics import average_precision_score
    from .io import read_jsonl
    core.aggregate(root)
    result = read_json(root / "results.json")
    rows = [r for shard in (0, 1) for r in read_jsonl(root / f"shard_{shard}/records.jsonl")]
    y = np.array([r["label"] for r in rows])
    reference = np.array([r["scores"]["current_only"] for r in rows])
    rng = np.random.default_rng(2026101905)
    groups = [np.flatnonzero(y == c) for c in (0, 1)]
    comparisons = []
    for arm in ARMS[2:]:
        scores = np.array([r["scores"][arm] for r in rows])
        delta = average_precision_score(y, scores) - average_precision_score(y, reference)
        boot = []
        for _ in range(5000):
            idx = np.concatenate([rng.choice(g, len(g), replace=True) for g in groups])
            boot.append(average_precision_score(y[idx], scores[idx]) - average_precision_score(y[idx], reference[idx]))
        exceed = 0
        for _ in range(10000):
            swap = rng.integers(0, 2, len(y)).astype(bool)
            a, b = np.where(swap, reference, scores), np.where(swap, scores, reference)
            exceed += abs(average_precision_score(y, a) - average_precision_score(y, b)) >= abs(delta) - 1e-12
        comparisons.append({"arm": arm, "AP_delta": float(delta), "ci95": np.quantile(boot, [.025, .975]).tolist(), "p": (exceed + 1) / 10001})
    previous = 0
    for i, c in enumerate(sorted(comparisons, key=lambda c: c["p"])):
        previous = max(previous, min(1., (len(comparisons) - i) * c["p"]))
        c["holm_p"] = previous
    result["versus_current_only"] = comparisons
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
        core.run(args.root, args.shard, bias_class=ContextBias)
    else:
        aggregate(args.root)


if __name__ == "__main__":
    main()
