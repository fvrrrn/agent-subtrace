"""Explicit causal inference functions: callers own models, prompts and outputs.

No module monkey-patching, globals, hidden CLI rewriting or experiment resumption.
These functions are infrastructure, not a new approved experimental protocol.
"""

from .chronology import prefixes
from .graphs import graph_summary, threshold_parents
from .rendering import render


def positive_probability(result, question_id, positive_key):
    for mapping, probabilities in zip(result["mapping"], result["native_probabilities"]):
        if mapping["id"] == question_id:
            index = 1 if mapping["type"] == "noul" else mapping["keys"].index(positive_key)
            return probabilities[index]
    raise ValueError(f"Missing question: {question_id}")


def infer_links(backend, case_id, model_input, question):
    points = []
    for prefix in prefixes(model_input):
        if not prefix.candidate_ids:
            continue
        questions = {
            f"link_{source}_{prefix.event_id}": {**question, "instructions": question["instructions"].format(source=source, target=prefix.event_id)}
            for source in prefix.candidate_ids
        }
        request = {"model": "kev-latest", "state": render(model_input.profile, prefix), "questions": questions}
        result = backend.predict(case_id, "link", request)
        points.extend({"source": source, "target": prefix.event_id,
                       "score": positive_probability(result, f"link_{source}_{prefix.event_id}", "LINKED")}
                      for source in prefix.candidate_ids)
    return points


def build_graphs(model_input, link_points, threshold):
    indexed = {}
    for point in link_points:
        pair = (point["source"], point["target"])
        if pair in indexed:
            raise ValueError("Duplicate LINK pair")
        indexed[pair] = point["score"]
    parents, consumed = {}, set()
    for prefix in prefixes(model_input):
        for source in prefix.candidate_ids:
            parents.setdefault(source, ())
        scores = {s: indexed[(s, prefix.event_id)] for s in prefix.candidate_ids}
        consumed.update((s, prefix.event_id) for s in prefix.candidate_ids)
        parents[prefix.event_id] = threshold_parents(prefix.candidate_ids, scores, threshold)
    if consumed != set(indexed):
        raise ValueError("LINK scores contain noncausal or unknown pairs")
    graph_summary(tuple(parents), parents)
    return parents


def safety_requests(model_input, prefix, graphs, question, notice):
    """graphs maps named arms to (presentation, full causal DAG)."""
    state = render(model_input.profile, prefix)
    requests = {"baseline": {"model": "kev-latest", "state": state, "questions": {"safety": question}}}
    ids = prefix.candidate_ids + (prefix.event_id,)
    for name, (style, graph) in graphs.items():
        if name == "baseline":
            raise ValueError("The baseline arm cannot be overridden by a graph")
        parents = {eid: graph[eid] for eid in ids}
        graph_summary(ids, parents)
        instructions = question["instructions"] + "\n\n" + notice
        if style == "list":
            instructions += "\n" + "\n".join(f"{eid}: tentative_parents=[{', '.join(sources)}]" for eid, sources in parents.items())
            rendered = state
        elif style == "inline":
            rendered = render(model_input.profile, prefix, parents)
        else:
            raise ValueError(f"Unknown graph presentation: {style}")
        requests[name] = {"model": "kev-latest", "state": rendered,
                          "questions": {"safety": {**question, "instructions": instructions}}}
    return requests


def infer_safety(backend, case_id, model_input, graphs, question, notice):
    steps = []
    for prefix in prefixes(model_input):
        requests = safety_requests(model_input, prefix, graphs, question, notice)
        # Reject an entire matched action before evaluating any arm.
        for request in requests.values():
            backend.validate(request)
        scores = {}
        for name, request in requests.items():
            result = backend.predict(case_id, name, request)
            scores[name] = positive_probability(result, "safety", "VIOLATION")
        steps.append({"event_id": prefix.event_id, "scores": scores})
    if not steps:
        raise ValueError("No proposed actions")
    return {"case_id": case_id, "steps": steps,
            "trajectory_scores": {name: max(step["scores"][name] for step in steps) for name in steps[0]["scores"]},
            "aggregation": "maximum pre-action score; not a calibrated trajectory probability"}
