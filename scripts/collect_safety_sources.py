"""Download pinned public source snapshots and provenance without executing repository code."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import tarfile
import time
import urllib.request
import urllib.parse

ROOT=Path('/var/local/danefedov/experiments/agent-safety/datasets/corpus-v1')
SOURCES=[
 ('hf','jc-ryan/AgentAlign','training_candidates','Aligned responses and benign tool traces; harmful requests often have SAFE refusals. Not binary action gold.'),
 ('hf','jinjinyien/ToolSafety','training_candidates','Direct/indirect harm and multistep refusal training. Request harmfulness is not action harmfulness.'),
 ('hf','AI45Research/AgentDoG1.0-Training-Data','training_candidates','Trajectory labels only; prefix-local reannotation required for step training.'),
 ('hf','LCZZZZ/HarnessAudit','generation_source','Task/tool/access-rule fixtures; not already labeled action trajectories.'),
 ('hf','thu-coai/Agent-SafetyBench','evaluation_or_reannotation','Check overlap with R-Judge, ToolEmu, AgentDojo and other source benchmarks.'),
 ('hf','nvidia/Nemotron-AIQ-Agentic-Safety-Dataset-1.0','evaluation_only','License restricts use to internal safety/security evaluation and benchmarking. Do not train or redistribute.'),
 ('hf','ninty-seven/stepguard_data','evaluation_only','Existing transfer benchmark. Never add this source or derivative scenarios to new train.'),
 ('hf','AI45Research/ATBench','evaluation_only','Existing public trajectory evaluation; no broadcast labels to actions.'),
 ('github','MurrayTom/ToolSafe','step_annotation_candidates','Published safe/unsafe/controversial step judgments; preserve official source splits and exclude 0.5 from binary gold.'),
 ('github','glo26/stepshield','step_annotation_candidates','Train/benign candidates only. Divergence timing is not an unsafe label for every later action; preserve canonical duplicate and holdout manifests.'),
 ('github','carriegu0818/StepSafe','step_annotation_candidates','Training annotations and intervention windows; CONTINUE/ASK/STOP differs from current-action violation. Inspect boundary semantics and source overlap.'),
 ('github','thu-coai/Agent-SafetyBench','trajectory_candidates','Inspect released 4000 scorer trajectories; no automatic unsafe prefix labels.'),
 ('github','Open-Agent-Safety/OpenAgentSafety','trajectory_candidates','Real-tool environments and published logs; download only, never execute tools or benchmark environments.'),
 ('github','ethz-spylab/agentdojo','evaluation_or_generation_source','Task fixtures and adversarial tool outputs; exposure does not establish unsafe agent action.'),
 ('github','ryoungj/ToolEmu','evaluation_or_generation_source','Tool schemas/scenarios and emulated outcomes; risk from outcomes must not leak into current-action inputs.'),
 ('github','SALT-NLP/PrivacyLens','evaluation_or_generation_source','Privacy contextual integrity cases; inspect released labels before conversion.'),
 ('github','zhiyuanc2001/ToolAlign','training_candidates','SFT and preference data; chosen/rejected mixes helpfulness, harmlessness and autonomy, not binary safety gold.'),
 ('github','Yunhao-Feng/AgentHazard','evaluation_or_reannotation','Computer-use execution traces and scenarios. Do not execute included payloads; task objectives are not action labels.'),
 ('hf','Nizba/agentguard-benchmark-v1','evaluation_or_generation_source','Inter-agent message injection labels, not agent-action safety labels.'),
 ('hf','responsible-ai-labs/rail-guard-benchmark','evaluation_only','Tool-call scenarios and content prompts. Keep source gold distinct from actual executed actions.'),
 ('hf','aryankeluskar/ToolAlignBench','evaluation_only','Competing deployment and safety constraints across regulated domains; policy conflicts require explicit scope.'),
]


def get(url):
    return urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'agent-safety-research/1.0'}),timeout=180)


def download(url,path):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():return
    for attempt in range(3):
        try:
            with get(url) as response,path.with_suffix(path.suffix+'.partial').open('wb') as f:
                while data:=response.read(1024*1024):f.write(data)
            path.with_suffix(path.suffix+'.partial').rename(path);return
        except Exception:
            if attempt==2:raise
            time.sleep(2**attempt)


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        while b:=f.read(1024*1024):h.update(b)
    return h.hexdigest()


def acquire(source):
    kind,repo,role,note=source;folder=ROOT/'raw'/repo.replace('/','--');folder.mkdir(parents=True,exist_ok=True)
    existing=folder/'SOURCE_MANIFEST.json'
    if existing.exists() and json.loads(existing.read_text()).get('kind')!=kind:
        folder=folder.with_name(folder.name+'--'+kind);folder.mkdir(parents=True,exist_ok=True)
    manifest=folder/'SOURCE_MANIFEST.json'
    if manifest.exists() and json.loads(manifest.read_text()).get('status')=='complete':return json.loads(manifest.read_text())
    result={'kind':kind,'repository':repo,'role':role,'note':note,'status':'started','retrieved_at_unix':time.time()}
    try:
        if kind=='hf':
            with get('https://huggingface.co/api/datasets/'+repo) as f:meta=json.load(f)
            revision=meta['sha'];result['revision']=revision;result['card_metadata']=meta.get('cardData',{})
            files=[x['rfilename'] for x in meta['siblings'] if x['rfilename'].lower().endswith(('.json','.jsonl','.parquet','.yaml','.yml','.md','.txt')) or Path(x['rfilename']).name.upper().startswith(('LICENSE','NOTICE','COPYING'))]
            if repo.startswith('nvidia/'):
                files=[f for f in files if '/traces/' not in f and not f.endswith('attack_manifest.jsonl')]
            def fetch(name):
                assert not Path(name).is_absolute() and '..' not in Path(name).parts
                download('https://huggingface.co/datasets/'+repo+'/resolve/'+revision+'/'+urllib.parse.quote(name),folder/name)
            with concurrent.futures.ThreadPoolExecutor(6) as pool:list(pool.map(fetch,files))
        else:
            with get('https://api.github.com/repos/'+repo+'/commits/HEAD') as f:meta=json.load(f)
            revision=meta['sha'];result['revision']=revision
            archive=folder/'snapshot.tar.gz';download('https://codeload.github.com/'+repo+'/tar.gz/'+revision,archive)
            files=[]
            with tarfile.open(archive) as tar:
                for member in tar:
                    parts=Path(member.name).parts[1:]
                    if not member.isfile() or not parts or '..' in parts:continue
                    name=Path(*parts)
                    if name.is_absolute():raise ValueError(name)
                    # Keep original source/data/docs; skip binary weights and images.
                    if name.suffix.lower() not in ('.json','.jsonl','.parquet','.yaml','.yml','.md','.txt','.py','.csv','.toml','.sh','.gz','.tar') and not name.name.upper().startswith(('LICENSE','NOTICE','COPYING')):continue
                    path=folder/'snapshot'/name;path.parent.mkdir(parents=True,exist_ok=True)
                    with tar.extractfile(member) as src,path.open('wb') as dst:
                        while data:=src.read(1024*1024):dst.write(data)
                    files.append(str(path.relative_to(folder)))
            result['archive_sha256']=digest(archive)
        result['files']=[{'path':f,'bytes':(folder/f).stat().st_size,'sha256':digest(folder/f)} for f in sorted(files)]
        result['status']='complete';result['bytes']=sum(x['bytes'] for x in result['files'])
    except Exception as e:result.update(status='failed',error=repr(e))
    manifest.write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(repo,result['status'],len(result.get('files',[])),result.get('error',''),flush=True)
    return result


def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    results=[]
    with concurrent.futures.ThreadPoolExecutor(4) as pool:
        for result in pool.map(acquire,SOURCES):
            results.append(result)
            (ROOT/'source_registry.json').write_text(json.dumps(results,ensure_ascii=False,indent=2))
    (ROOT/'collection_status.json').write_text(json.dumps({'complete':sum(r['status']=='complete' for r in results),
        'failed':sum(r['status']=='failed' for r in results),'source_count':len(results),'stage':'raw_acquisition_not_training_ready'},indent=2))


if __name__=='__main__':main()
