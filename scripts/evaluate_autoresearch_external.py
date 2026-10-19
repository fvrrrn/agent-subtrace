"""Final-only protected replay for a locked autoresearch finalist; never fit thresholds."""
import argparse
import json
import math
from pathlib import Path
import time
import traceback

import torch
from tqdm.auto import tqdm

import train_whole_dialogs as original
import train_autoresearch_heads as trainer
from train_curve_event_heads import Head
from prepare_whole_dialogs import BASE, REVISION, digest, write
from graph_safety.metrics import evaluate


def checked_file(entry):
    path = Path(entry['path'])
    assert original.file_hash(path) == entry['sha256'], f'Hash mismatch: {path}'
    return path


def summarize(rows, threshold):
    assert rows
    labels = [r['label'] for r in rows]
    scores = [r['score'] for r in rows]
    output = {}
    for name, value in (('dev_select_threshold', threshold), ('control_0.5', .5)):
        metrics = evaluate(labels, scores, threshold=value)
        metrics.pop('pr_curve')
        specificity = metrics['operating_point']['specificity']
        metrics['operating_point']['fpr'] = 1-specificity if specificity is not None else None
        output[name] = metrics
    output['class_scores'] = {}
    q = [0., .05, .25, .5, .75, .95, 1.]
    for label in (0, 1):
        values = torch.tensor([r['score'] for r in rows if r['label'] == label], dtype=torch.float64)
        output['class_scores'][str(label)] = {'count': len(values), 'quantile_levels': q,
            'quantiles': torch.quantile(values, values.new_tensor(q)).tolist() if len(values) else None,
            'fraction_ge_half': (values >= .5).double().mean().item() if len(values) else None,
            'fraction_ge_dev_threshold': (values >= threshold).double().mean().item() if len(values) else None}
    output['ids'] = [r['key'] for r in rows]
    return output


def report(predictions, suite, threshold, common_keys=None):
    keys = [r['key'] for r in predictions]
    assert len(keys) == len(set(keys))
    reports = {}
    cohorts = {'all_fully_evaluated': predictions}
    if common_keys is not None:
        assert common_keys and len(common_keys) == len(set(common_keys))
        assert set(common_keys) <= set(keys), 'Common report IDs must all be fully evaluated'
        cohorts['explicit_common_ids'] = [r for r in predictions if r['key'] in set(common_keys)]
    for cohort, cases in cohorts.items():
        if suite == 'dojo':
            assert all(len(r['actions']) == 1 for r in cases), 'Dojo requires one gold action per case'
            reports[cohort] = {'action_gold': summarize([
                {'key': r['key'], 'label': r['label'], 'score': r['actions'][0]['score']} for r in cases], threshold)}
            continue
        modes = {}
        for mode in ('all_events', 'tool_only', 'all_events_on_tool_cases'):
            selected = []
            for case in cases:
                tool_actions = [a for a in case['actions'] if a['kind'] == 'tool_call']
                if mode != 'all_events' and not tool_actions:
                    continue
                actions = tool_actions if mode == 'tool_only' else case['actions']
                selected.append({'key': case['key'], 'label': case['label'],
                                 'score': max(a['score'] for a in actions)})
            modes[mode] = summarize(selected, threshold) if selected else {'count': 0, 'ids': []}
        reports[cohort] = modes
    return reports


def run(args):
    # One locked finalist and one suite per invocation/GPU. No resume/retry.
    plan = original.read(args.plan)
    out = args.root / args.suite
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    completed = []
    current_case = current_sequence = None
    expected_keys = []
    observed_max_sequence_seconds = 0.
    try:
        assert plan['final_only'] is True
        assert plan['mode'] in ('frozen', 'lora', 'online_frozen', 'fullft')
        assert plan['readout'] in ('endpoint', 'linear_softmax', 'null', 'multiscale', 'leaky')
        assert isinstance(plan['attention'], bool)
        write(out / 'plan.json', plan)
        prompt = plan.get('prompt')
        if prompt is not None:
            from prepare_autoresearch_prompt import INSTRUCTION
            import hashlib
            assert plan['mode'] == 'online_frozen', 'Prompt branch requires explicit new online encoding'
            assert prompt['text'] == INSTRUCTION
            assert prompt['sha256'] == hashlib.sha256(INSTRUCTION.encode()).hexdigest()
            training_audit = original.read(checked_file(prompt['training_audit']))
            assert training_audit['prompt'] == prompt['text'] and training_audit['prompt_sha256'] == prompt['sha256']
        threshold = float(plan['threshold'])
        assert math.isfinite(threshold) and 0 <= threshold <= 1
        assert plan['threshold_source']['split'] == 'dev_select'
        threshold_record = original.read(checked_file(plan['threshold_source']))
        assert threshold_record['split'] == 'dev_select' and float(threshold_record['threshold']) == threshold
        deadline = float(plan['deadline_unix'])
        reserve = float(plan.get('deadline_reserve_seconds', 120))
        first_sequence = float(plan.get('sequence_budget_seconds', 180))
        assert math.isfinite(deadline) and math.isfinite(reserve) and reserve >= 0
        assert math.isfinite(first_sequence) and first_sequence > 0

        def check_deadline():
            remaining = deadline-time.time()
            needed = reserve+max(first_sequence, 1.5*observed_max_sequence_seconds)
            if remaining <= needed:
                raise TimeoutError(f'Deadline guard: {remaining:.1f}s remaining, {needed:.1f}s reserved; no case silently omitted')

        check_deadline()
        inputs = plan['suites'][args.suite]
        cases_path = checked_file(inputs['cases'])
        spans_path = checked_file(inputs['spans'])
        audit_path = checked_file(inputs['audit'])
        rejections_path = checked_file(inputs['shared_rejections'])
        audit = original.read(audit_path)
        assert audit['cases_sha256'] == inputs['cases']['sha256']
        assert audit['template_sha256'] == plan['template_sha256'] and audit['no_truncation']
        if prompt is not None:
            assert audit['prompt'] == prompt['text'] and audit['prompt_sha256'] == prompt['sha256']
            assert audit['new_exclusions'] == [] and audit['preparation_only']
            assert audit['tokenizer_vocab_sha256'] == training_audit['tokenizer_vocab_sha256']
        else:
            assert 'prompt_sha256' not in audit, 'Prompted inputs require explicit prompt plan'
        cases = original.read(cases_path)
        spans = original.read(spans_path)
        expected_keys = [c['key'] for c in cases]
        assert cases and len(expected_keys) == len(set(expected_keys))
        assert len(cases) == audit['admitted']
        inherited_rejections = original.read(rejections_path)
        assert len(inherited_rejections) == audit['rejected']
        write(out / 'inherited_exclusions.json', {'audit': audit, 'rejections': inherited_rejections,
              'new_context_exclusions': [], 'policy': 'Use unchanged protected admitted subset; no truncation or new exclusions'})
        for case in cases:
            assert case['label'] in (0, 1) and case['actions']
            assert case['ids_sha256'] == digest([s['ids_sha256'] for s in case['sequences']])
            for seq in case['sequences']:
                assert seq['tokens'] == len(seq['ids']) <= 65536
                assert seq['ids_sha256'] == digest(seq['ids'])
                assert len(seq['action_indices']) == len(seq['endpoints']) > 0
        head_path = checked_file(plan['head'])
        fullft_path = checked_file(plan['backbone']) if plan['mode'] == 'fullft' else None
        adapter_files = None
        if plan['mode'] == 'lora':
            adapter = Path(plan['adapter']['path'])
            adapter_files = {str(p.relative_to(adapter)): original.file_hash(p)
                             for p in sorted(adapter.rglob('*')) if p.is_file()}
            assert adapter_files and adapter_files == plan['adapter']['files_sha256']
        sources = {Path(module.__file__).name: original.file_hash(Path(module.__file__))
                   for module in (trainer, original)}
        sources[Path(__file__).name] = original.file_hash(Path(__file__))
        import train_curve_event_heads
        sources[Path(train_curve_event_heads.__file__).name] = original.file_hash(Path(train_curve_event_heads.__file__))
        trainer.seed(int(plan.get('seed', original.SEED)))
        head = Head('curve_null' if plan['readout'] == 'null' else 'curve_linear_softmax', plan['attention']).cuda()
        head.load_state_dict(torch.load(head_path, map_location='cpu', weights_only=True))
        head.eval().requires_grad_(False)
        model = None
        if plan['mode'] in ('lora', 'online_frozen', 'fullft'):
            check_deadline()
            model = original.backbone(False)
            assert original.frozen_hash(model) == plan['frozen_weights_sha256'], 'Pinned base weight mismatch'
            if plan['mode'] == 'lora':
                from peft import PeftModel
                model = PeftModel.from_pretrained(model, adapter, is_trainable=False)
            elif plan['mode'] == 'fullft':
                model.float()
                model.load_state_dict(torch.load(fullft_path, map_location='cpu', weights_only=True), strict=True)
                for parameter in model.parameters():
                    parameter.data = parameter.data.to(torch.bfloat16)
            model.eval().requires_grad_(False)
        environment = trainer.environment()
        write(out / 'protocol.json', {'plan_sha256': original.file_hash(args.plan), 'sources': sources,
              'head_sha256': plan['head']['sha256'], 'adapter_files_sha256': adapter_files,
              'trained_backbone_sha256': plan['backbone']['sha256'] if fullft_path is not None else None,
              'fullft_forward_precision': 'FP32 checkpoint loaded first; BF16 parameter copies, FP32 buffers, BF16 autocast' if fullft_path is not None else None,
              'base': BASE, 'revision': REVISION, 'environment': environment,
              'threshold': threshold, 'threshold_fit': False, 'actual_input_cases': len(cases),
              'primary': 'Dojo action gold; trajectory max over all events INCLUDING Complete otherwise',
              'forward_unit': 'unchanged native stable whole dialog or recorded causal-prefix fallback',
              'no_truncation': True, 'prompt': prompt})
        (out / 'tokens').mkdir()
        torch.cuda.reset_peak_memory_stats()
        total_tokens = total_sequences = 0
        with (out / 'predictions.jsonl').open('w') as stream, torch.inference_mode():
            for case in tqdm(sorted(cases, key=lambda c: (-c['tokens'], c['key'])), desc=args.suite, unit='case'):
                current_case = case['key']
                check_deadline()
                case_started = time.monotonic()
                results = {}
                for seq in case['sequences']:
                    current_sequence = seq['key']
                    check_deadline()
                    seq_started = time.monotonic()
                    if model is None:
                        cache = Path(inputs['cache']) / (digest(seq['key'])+'.pt')
                        with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
                            saved = torch.load(cache, map_location='cpu', weights_only=True)
                        expected_identity = {'ids_sha256': seq['ids_sha256'], 'revision': REVISION,
                            'template_sha256': plan['template_sha256'], 'torch': str(torch.__version__),
                            'attention': 'sdpa', 'dtype': 'bf16', 'padding': 'one sequence, right to128',
                            'weights_sha256': plan['frozen_weights_sha256']}
                        assert saved['identity'] == expected_identity, f'Cache identity mismatch: {cache}'
                        hidden = saved['hidden'].to('cuda')
                        del saved
                    else:
                        if plan['mode'] == 'fullft':
                            with torch.autocast('cuda', dtype=torch.bfloat16):
                                hidden = original.encode(model, seq)
                        else:
                            hidden = original.encode(model, seq)
                    assert hidden.shape == (1, math.ceil(seq['tokens']/128)*128, 2560)
                    assert hidden.dtype == torch.bfloat16 and torch.isfinite(hidden).all()
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        raw, null = head(hidden)
                    z = raw[0, :seq['tokens']].float()
                    n = null[0, :seq['tokens']].float() if null is not None else None
                    assert torch.isfinite(z).all() and (n is None or torch.isfinite(n).all())
                    action_spans = []
                    for index, endpoint in zip(seq['action_indices'], seq['endpoints']):
                        assert index not in results
                        action = case['actions'][index]
                        span = spans[case['key']][index]
                        assert action['key'] == span['key'] and action['state_sha256'] == span['state_sha256']
                        assert 0 <= span['start'] < span['end'] == endpoint+1 <= seq['tokens']
                        assert digest(seq['ids'][:span['end']]) == span['prefix_ids_sha256']
                        event = trainer.event_logit(z, n, span, plan['readout'])
                        assert torch.isfinite(event)
                        results[index] = {'key': action['key'], 'kind': action['kind'],
                            'state_sha256': action['state_sha256'], 'event_logit': event.item(),
                            'score': event.sigmoid().item(), 'action_tokens': span['end']-span['start'],
                            'null_logit': n[endpoint].item() if n is not None else None}
                        action_spans.append(span)
                    torch.cuda.synchronize()
                    seq_seconds = time.monotonic()-seq_started
                    observed_max_sequence_seconds = max(observed_max_sequence_seconds, seq_seconds)
                    torch.save({'case': case['key'], 'sequence': seq['key'],
                        'ids': torch.tensor(seq['ids'], dtype=torch.int32), 'ids_sha256': seq['ids_sha256'],
                        'raw_logits': z.cpu(), 'null_logits': n.cpu() if n is not None else None,
                        'action_indices': seq['action_indices'], 'action_spans': action_spans,
                        'actual_tokens': seq['tokens'], 'padded_tokens': hidden.shape[1],
                        'seconds': seq_seconds, 'head_sha256': plan['head']['sha256']},
                        out / 'tokens' / (digest([case['key'], seq['key']])+'.pt'))
                    total_tokens += seq['tokens']; total_sequences += 1
                    del hidden, raw, null, z, n, event
                assert sorted(results) == list(range(len(case['actions'])))
                prediction = {'key': case['key'], 'label': case['label'], 'ids_sha256': case['ids_sha256'],
                    'actions': [results[i] for i in sorted(results)], 'actual_tokens': case['tokens'],
                    'evaluation_mode': case['evaluation_mode'], 'seconds': time.monotonic()-case_started}
                stream.write(json.dumps(prediction)+'\n')
                stream.flush()
                completed.append(prediction)
        assert {r['key'] for r in completed} == set(expected_keys)
        common_keys = None
        if 'common_ids' in inputs:
            common_keys = original.read(checked_file(inputs['common_ids']))
        summary = report(completed, args.suite, threshold, common_keys)
        write(out / 'summary.json', {'results': summary, 'fully_evaluated_ids': sorted(expected_keys),
              'new_exclusions': [], 'inherited_exclusion_count': len(inherited_rejections),
              'limitations': 'Final report only, not adaptive selection. ATBench/R-Judge trajectory gold is not action gold. Previously viewed protected benchmarks; inherited subset and causal replay differ from official completed-trace protocols.'})
        write(out / 'complete.json', {'status': 'complete', 'cases': len(completed),
              'sequences': total_sequences, 'encoded_or_replayed_tokens': total_tokens,
              'seconds': time.monotonic()-started, 'peak_cuda_bytes': torch.cuda.max_memory_allocated(),
              'qwen_loaded': model is not None, 'errors': 0, 'new_exclusions': 0,
              'predictions_sha256': original.file_hash(out / 'predictions.jsonl')})
    except BaseException as error:
        write(out / 'failure.json', {'status': 'NOT_IMPLEMENTED' if isinstance(error, NotImplementedError) else 'failed',
              'error': repr(error), 'traceback': traceback.format_exc(), 'case': current_case,
              'sequence': current_sequence, 'completed_ids': [r['key'] for r in completed],
              'incomplete_or_unstarted_ids': sorted(set(expected_keys)-{r['key'] for r in completed}),
              'seconds': time.monotonic()-started, 'no_automatic_retry': True,
              'partial_outputs_preserved': True, 'valid_complete_benchmark': False})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--suite', choices=('atbench', 'rjudge', 'dojo'), required=True)
    run(parser.parse_args())
