"""Deadline-bounded six conditions; child tqdm is streamed to this shell."""
from pathlib import Path
import time
import traceback
from tqdm import tqdm
import launch_frozen_instruct_4b_v2 as pipeline
from train_streaming_weighted_multilayer_v1 import RUN, DEADLINE

pipeline.RUN = RUN


def main():
    assert time.time() + 1500 < DEADLINE, 'Insufficient remaining budget for all six conditions'
    source = Path(__file__).parent
    manifest = {str(p.relative_to(source)): pipeline.file_hash(p) for p in source.rglob('*')
                if p.is_file() and '__pycache__' not in p.parts}
    pipeline.write(source.parent / 'source_manifest.json', manifest)
    pipeline.write(RUN / 'authorization.json', {'conditions': [f'{features}_{curve}'
        for features in ('last', 'three') for curve in ('step', 'linear', 'broad')],
        'epochs_each': 2, 'training_deadline': DEADLINE, 'overall_deadline': 1792348785,
        'frozen': True, 'lora': False, 'added_attention': False, 'threshold': .5,
        'selection': 'AP; tie lower BCE; earliest epoch',
        'source_manifest_sha256': pipeline.file_hash(source.parent / 'source_manifest.json'),
        'protocol_sha256': pipeline.file_hash(source / 'STREAMING_PROTOCOL.md')})
    trainer = 'train_streaming_weighted_multilayer_v1.py'
    with tqdm(total=15, desc='Six weighted/multilayer conditions', unit='stage') as progress:
        pipeline.run_stage('last_cpu_admission', trainer, 'cpu', '--feature-set', 'last', gpu=False)
        progress.update()
        pipeline.run_stage('three_cache', 'cache_streaming_weighted_multilayer_v1.py')
        progress.update()
        pipeline.run_stage('three_cpu_admission', trainer, 'cpu', '--feature-set', 'three', gpu=False)
        progress.update()
        for features in ('last', 'three'):
            for curve in ('step', 'linear', 'broad'):
                for phase in ('gate', 'train'):
                    pipeline.run_stage(f'{features}_{curve}_{phase}', trainer, phase,
                        '--feature-set', features, '--curve', curve)
                    progress.update()
                assert pipeline.read(RUN / f'{features}_{curve}/fit/complete.json')['epochs'] == 2
    pipeline.write(RUN / 'status.json', {'status': 'complete', 'conditions': 6, 'epochs_each': 2})


if __name__ == '__main__':
    try:
        main()
    except Exception:
        failure = {'status': 'failed', 'traceback': traceback.format_exc()}
        pipeline.write(RUN / f'failure_{time.time_ns()}.json', failure)
        pipeline.write(RUN / 'status.json', failure)
        raise
