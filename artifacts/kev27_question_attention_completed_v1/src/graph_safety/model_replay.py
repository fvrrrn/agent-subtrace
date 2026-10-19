"""Matched 27B replay of the two frozen graph pilots; recompute all model scores."""

import argparse
from pathlib import Path
import json
import random
import time

from . import oracle_pilot as oracle
from . import reverse_score_pilot as reverse
from .chronology import prefixes
from .inference import KevBackend
from .io import read_json, read_jsonl, sha256, write_json
from .rendering import render
from .runner import positive_probability
from .schema import ModelInput

CHECKPOINT = {
    "repo": "jaredpalmer/kev-27b", "revision": "af0e6d551bdc2cc724f3e9d7a8bee1cd4fb8f7bf",
    "weights_sha": "d27af6ab2be16824166ac639907b4dba40979ff338599c2872721aa6c5072022",
    "head_sha": "7968f17b03479c1ef9d1c0f3ab8a15e31ecb441cf40691b07ee945ab554d45ad",
    "base": "Qwen/Qwen3.8-27B", "base_revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
    "source_commit": "85ccf56a0afb952323c621055b02876e8498d60c",
}


def prepare(root, oracle_source, reverse_source):
    root.mkdir(parents=True, exist_ok=False)
    for suite, source in (("oracle",oracle_source),("reverse",reverse_source)):
        destination=root/suite;destination.mkdir()
        original=read_json(source/"protocol.json")
        for split in ("dev","locked"):
            path=source/f"{split}_cases.json"
            if sha256(path)!=original["data_sha256"][split]:raise ValueError("Original data hash mismatch")
            (destination/path.name).write_bytes(path.read_bytes())
        protocol={**original,"checkpoint":CHECKPOINT,"model_replay":"Same examples and prompts as 4B; previously viewed evaluation data; exploratory paired model comparison.",
            "reference_protocol_sha256":sha256(source/"protocol.json"),
            "reference_report_sha256":sha256(source/"locked_report.json"),
            "parity_gate":"No invalid comparison of 27B against 4B probabilities. Check native two-pass repeat on the same fixed request, maximum absolute difference <= .001 and identical .5 decision. This is numerical repeatability, not full backend parity.",
            "prior_scores":"Recomputed from 27B causal baseline; never reuse 4B scores.",
            "code_sha256":{p.name:sha256(p) for p in Path(__file__).parent.glob("*.py")}}
        write_json(destination/"protocol.json",protocol)
        write_json(destination/"reference_4b_report.json",read_json(source/"locked_report.json"))


def run(root,suite,split,shard):
    from tqdm.auto import tqdm

    module=oracle if suite=="oracle" else reverse
    protocol=read_json(root/"protocol.json")
    for name,expected in protocol["code_sha256"].items():
        if sha256(Path(__file__).parent/name)!=expected:raise ValueError("Code changed")
    if sha256(root/f"{split}_cases.json")!=protocol["data_sha256"][split]:raise ValueError("Data changed")
    if split=="locked" and not (root/"selection.json").exists():raise ValueError("Dev threshold not frozen")
    folder=root/f"{split}_{shard}";folder.mkdir(exist_ok=False)
    summary={"status":"loading","split":split,"shard":shard,"attempted":0,"checkpoint":CHECKPOINT,"protocol_sha256":sha256(root/"protocol.json")}
    started=time.perf_counter()
    try:
        with KevBackend(CHECKPOINT,folder/"calls.jsonl") as backend:
            summary["environment"]=backend.environment
            cases=read_json(root/f"{split}_cases.json")
            request=oracle.requests(cases[0])["baseline"]
            a=backend.predict("REPEATABILITY","first",request)
            b=backend.predict("REPEATABILITY","repeat",request)
            delta=max(abs(x-y) for aa,bb in zip(a["native_probabilities"],b["native_probabilities"]) for x,y in zip(aa,bb))
            same=(positive_probability(a,"safety","VIOLATION")>=.5)==(positive_probability(b,"safety","VIOLATION")>=.5)
            write_json(folder/"repeatability.json",{"max_difference":delta,"same_decision":same})
            if delta>.001 or not same:raise ValueError("Repeatability gate failed")
            summary["status"]="running";write_json(folder/"summary.json",summary)
            selected = [case for index, case in enumerate(cases) if (index // 2) % 2 == shard]
            for case in tqdm(selected, desc=f"27B/{suite}/{split}/shard-{shard}", unit="trajectory", mininterval=5):
                extra={}
                if suite=="reverse":
                    model_input=ModelInput.from_legacy(case);prior={};prior_calls={}
                    for prefix in prefixes(model_input):
                        if prefix.event_id==case["target_event"]:break
                        req={"model":"kev-latest","state":render(model_input.profile,prefix),"questions":{"safety":oracle.SAFETY}}
                        prediction=backend.predict(case["key"],f"prior:{prefix.event_id}",req)
                        prior[prefix.event_id]=positive_probability(prediction,"safety","VIOLATION")
                        prior_calls[prefix.event_id]=prediction["call"]
                    requests,shuffled=reverse.build_requests(case,prior)
                    extra={"prior_scores":prior,"prior_calls":prior_calls,"shuffled_scores":shuffled,
                           "shuffled_display_values_changed":sum(f"{prior[e]:.4f}"!=f"{s:.4f}" for e,s in shuffled.items())}
                else:requests=oracle.requests(case)
                for req in requests.values():backend.validate(req)
                order=list(module.ARMS);random.Random(case["key"]).shuffle(order)
                results={arm:backend.predict(case["key"],arm,requests[arm]) for arm in order}
                row={"key":case["key"],"group":case["group"],"family":case["family"],"label":case["label"],"input_sha256":case["input_sha256"],
                     "scores":{arm:positive_probability(r,"safety","VIOLATION") for arm,r in results.items()},"results":results,"order":order,**extra}
                with (folder/"records.jsonl").open("a",encoding="utf-8") as stream:stream.write(json.dumps(row,ensure_ascii=False)+"\n")
                summary["attempted"]+=1;write_json(folder/"summary.json",summary)
            summary["status"]="complete"
    except BaseException as error:
        summary.update(status="failed_or_interrupted",error=f"{type(error).__name__}: {error}")
        raise
    finally:
        summary["seconds"]=time.perf_counter()-started;write_json(folder/"summary.json",summary)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase",choices=("prepare","run","select","aggregate"));parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--oracle-source",type=Path);parser.add_argument("--reverse-source",type=Path)
    parser.add_argument("--suite",choices=("oracle","reverse"));parser.add_argument("--split",choices=("dev","locked"));parser.add_argument("--shard",type=int,choices=(0,1))
    args=parser.parse_args()
    if args.phase=="prepare":prepare(args.root,args.oracle_source,args.reverse_source);return
    module=oracle if args.suite=="oracle" else reverse
    if args.phase=="run":run(args.root,args.suite,args.split,args.shard)
    elif args.phase=="select":module.select(args.root)
    else:module.aggregate(args.root)


if __name__=="__main__":main()
