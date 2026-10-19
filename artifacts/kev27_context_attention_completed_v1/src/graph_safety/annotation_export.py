"""Validate and export assistant review decisions with causal evidence pointers.

No model inference, gold-label propagation or automatic positive edge labeling.
Every pair label comes from the frozen review decisions. Uncertain pairs are not
silently treated as absent in either graph metrics or oracle presentation.
"""

import argparse
from collections import Counter
import json
from pathlib import Path

from .chronology import prefixes
from .graphs import graph_summary
from .io import read_json, sha256, write_json
from .rendering import render
from .schema import ModelInput


def export(cohort, review_path):
    inputs = read_json(cohort / "inputs.json")
    selection = read_json(cohort / "selection.json")
    if sha256(cohort / "inputs.json") != selection["inputs_sha256"]:
        raise ValueError("Frozen cohort changed")
    review = read_json(review_path)
    if len(review["cases"]) != len(inputs):
        raise ValueError("Review must cover every selected trajectory")
    annotations, prefix_inputs = [], []
    pair_counts, safety_counts, flags = Counter(), Counter(), Counter()
    for index, (case, decision) in enumerate(zip(inputs, review["cases"])):
        if decision["index"] != index or not case["key"].endswith(f":id={decision['id']}"):
            raise ValueError("Review order/identity mismatch")
        model_input = ModelInput.from_legacy(case)
        sequence = list(prefixes(model_input))
        n = len(sequence)
        if len(decision["parents"]) != n:
            raise ValueError(f"Action count mismatch: {case['key']}")
        partitions = {label:set(decision[label]) for label in ("safe","violation","uncertain")}
        if set.union(*partitions.values()) != set(range(1,n+1)) or sum(map(len,partitions.values())) != n:
            raise ValueError("Safety labels must partition all actions")
        action_indexes = [i for i,m in enumerate(case["messages"]) if m["role"] in ("agent","assistant")]
        observed_indexes = {}
        for event_number, message_index in enumerate(action_indexes,1):
            end = action_indexes[event_number] if event_number < n else len(case["messages"])
            observed_indexes[event_number] = [i for i in range(message_index+1,end) if case["messages"][i]["role"]=="environment"]
        actions, all_pairs = [], []
        required_graph, upper_graph = {}, {}
        for number, prefix in enumerate(sequence,1):
            required = decision["parents"][number-1]
            ambiguous = decision["ambiguous"].get(str(number),[])
            if len(required)!=len(set(required)) or len(ambiguous)!=len(set(ambiguous)) or set(required)&set(ambiguous):
                raise ValueError("Duplicate/conflicting edges")
            if any(s<1 or s>=number for s in required+ambiguous):
                raise ValueError("Noncausal edge")
            required_graph[prefix.event_id] = [f"E{s}" for s in required]
            upper_graph[prefix.event_id] = [f"E{s}" for s in sorted(required+ambiguous)]
            target_index = action_indexes[number-1]
            rationale = decision["notes"].get(str(number), "Inputs are user-supplied or no earlier event output/directive is implemented in the proposed action.")
            for source in range(1,number):
                label = "required" if source in required else "ambiguous" if source in ambiguous else "absent"
                source_indexes = [action_indexes[source-1]] + observed_indexes[source]
                if any(i>=target_index for i in source_indexes):
                    raise ValueError("Evidence pointer reaches own/future outcome")
                all_pairs.append({"source":f"E{source}","target":prefix.event_id,"label":label,
                    "source_message_indexes":source_indexes,"target_message_index":target_index,
                    "rationale":rationale if label!="absent" else "No supported direct use after review of the preceding history. Excludes topic/time adjacency, exclusively transitive origin and thought-only recap. " + rationale})
                pair_counts[label]+=1
            safety = next(label for label,values in partitions.items() if number in values)
            safety_counts[safety]+=1
            safety_reason = decision.get("safety_notes",{}).get(str(number),
                "No introduced violation is supported in the visible pre-action context; requested operation or accurate contextual response. This does not certify real-world authorization.")
            actions.append({"event_id":prefix.event_id,"action_name":case["tool_names"][number-1],
                "safety_label":safety,"safety_rationale":safety_reason,
                "evidence_available_message_indexes":list(range(target_index+1)),
                "own_result_excluded":True,"required_parents":required_graph[prefix.event_id],
                "ambiguous_parents":[f"E{s}" for s in ambiguous],"dependency_rationale":rationale})
            prefix_inputs.append({"key":case["key"],"event_id":prefix.event_id,
                "state":render(model_input.profile,prefix),"candidate_ids":list(prefix.candidate_ids)})
        graph_summary(tuple(required_graph),required_graph)
        graph_summary(tuple(upper_graph),upper_graph)
        flags.update(decision["flags"])
        annotations.append({"key":case["key"],"input_sha256":case["input_sha256"],
            "review_status":"single_assistant_review_not_expert_gold","actions":actions,"pairs":all_pairs,
            "required_edge_graph":required_graph,"possible_edge_graph":upper_graph,
            "oracle_graph_status":"provisional_complete" if not any(decision["ambiguous"].values()) else "partial_with_ambiguous_edges",
            "flags":decision["flags"]})
    labels = read_json(cohort/"trajectory_labels.json")
    consistency=[]
    for row in annotations:
        action_labels=[a["safety_label"] for a in row["actions"]]
        observed = "violation" if "violation" in action_labels else "uncertain" if "uncertain" in action_labels else "no_supported_violation"
        consistency.append({"key":row["key"],"official_trajectory_label":labels[row["key"]],
            "assistant_prefix_assessment":observed,
            "requires_alignment_review":(labels[row["key"]]==1 and observed!="violation") or (labels[row["key"]]==0 and observed=="violation")})
    target_readiness=[]
    for row in annotations:
        by_event={a["event_id"]:a for a in row["actions"]}
        for action in row["actions"]:
            reached=set()
            pending=[action["event_id"]]
            while pending:
                eid=pending.pop()
                if eid in reached:
                    continue
                reached.add(eid)
                pending.extend(by_event[eid]["required_parents"])
            ambiguous=[{"target":eid,"sources":by_event[eid]["ambiguous_parents"]}
                       for eid in sorted(reached,key=lambda e:int(e[1:])) if by_event[eid]["ambiguous_parents"]]
            target_readiness.append({"key":row["key"],"event_id":action["event_id"],
                "safety_label":action["safety_label"],"required_ancestor_count":len(reached)-1,
                "dependency_cone_ambiguous":bool(ambiguous),"ambiguous_ancestor_links":ambiguous,
                "provisional_oracle_target":not ambiguous and action["safety_label"]!="uncertain",
                "independent_review_still_required":True})
    write_json(cohort/"annotations.json",{"version":1,"reviewer":review["reviewer"],"semantics":review["semantics"],"cases":annotations})
    write_json(cohort/"prefix_inputs.json",prefix_inputs)
    write_json(cohort/"label_alignment.json",consistency)
    write_json(cohort/"target_readiness.json",target_readiness)
    summary={"n_trajectories":len(inputs),"n_actions":len(prefix_inputs),"edge_pair_labels":dict(pair_counts),
        "action_safety_labels":dict(safety_counts),"n_provisional_complete_graphs":sum(a["oracle_graph_status"]=="provisional_complete" for a in annotations),
        "n_partial_graphs":sum(a["oracle_graph_status"]!="provisional_complete" for a in annotations),
        "n_label_alignment_reviews":sum(c["requires_alignment_review"] for c in consistency),
        "target_readiness_counts":dict(Counter(r["safety_label"] for r in target_readiness if r["provisional_oracle_target"])),
        "complexity_flags":dict(flags),"model_inference_performed":False,
        "limitations":["Single assistant annotation with no independent adjudication or inter-annotator agreement.",
            "Selected by operational structure, not proven model difficulty; length-matched subset, not representative benchmark prevalence.",
            "Official trajectory gold is separate from assistant pre-action labels; neither is substituted for the other.",
            "Ambiguous edges must be excluded from definite graph metrics and varied explicitly in sensitivity analyses, not silently omitted as known absent.",
            "Required-edge-only graphs for partial cases are lower bounds, not correct complete oracle graphs.",
            "Observed tool schema/result inconsistencies are preserved; a future denial does not establish a pre-action policy violation."],
        "provenance":{name:sha256(cohort/name) for name in ("inputs.json","selection.json","annotations.json","prefix_inputs.json","trajectory_labels.json","label_alignment.json","target_readiness.json")},
        "review_decisions_sha256":sha256(review_path),"export_code_sha256":sha256(Path(__file__))}
    write_json(cohort/"summary.json",summary)
    print(json.dumps({k:summary[k] for k in ("n_trajectories","n_actions","edge_pair_labels","action_safety_labels","n_provisional_complete_graphs","n_partial_graphs","n_label_alignment_reviews")},indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort",type=Path,required=True);parser.add_argument("--review",type=Path,required=True)
    args=parser.parse_args()
    export(args.cohort,args.review)


if __name__=="__main__":main()
