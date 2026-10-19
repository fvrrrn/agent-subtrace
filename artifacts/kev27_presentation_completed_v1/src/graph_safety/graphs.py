"""Binary temporal graphs; density is descriptive, never a correctness target."""

import math


def threshold_parents(candidate_ids, scores, threshold):
    if len(set(candidate_ids)) != len(candidate_ids) or set(scores) != set(candidate_ids):
        raise ValueError("Each preceding event must have exactly one LINK score")
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("LINK threshold must lie in [0, 1]")
    if any(not math.isfinite(s) or not 0 <= s <= 1 for s in scores.values()):
        raise ValueError("Invalid LINK probability")
    return tuple(sid for sid in candidate_ids if scores[sid] >= threshold)


def graph_summary(event_ids, parents):
    positions = {eid: i for i, eid in enumerate(event_ids)}
    if len(positions) != len(event_ids) or set(parents) != set(event_ids):
        raise ValueError("Graph must name each event exactly once")
    for target, sources in parents.items():
        if len(set(sources)) != len(sources):
            raise ValueError("Duplicate edge")
        if any(s not in positions or positions[s] >= positions[target] for s in sources):
            raise ValueError("Edges must point from earlier to later events")
    n = len(event_ids)
    edges = sum(map(len, parents.values()))
    possible = n * (n - 1) // 2
    return {"n_events": n, "n_edges": edges, "n_candidate_pairs": possible,
            "density": edges / possible if possible else None,
            "mean_indegree": edges / n if n else None}
