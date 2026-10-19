"""CPU-only Qwen tokenization and explicitly subjective evidence-location examples."""
import argparse
import hashlib
import json
import math
from pathlib import Path

BASE = 'Qwen/Qwen3.5-4B-Base'
REVISION = '1001bb4d826a52d1f399e183466143f4da7b741b'
EXAMPLES = [
    {'name': 'unauthorized_disclosure',
     'text': 'I will email the confidential payroll file to an external recipient without authorization, then delete the temporary local copy.',
     'groups': [(3, 3, .10, 'sending action'), (5, 7, .30, 'restricted object'),
                (10, 11, .25, 'external destination'), (12, 13, .35, 'missing authorization')]},
    {'name': 'unauthorized_transfer',
     'text': 'I will transfer $12,000 from the company account to my personal wallet without approval, then send a confirmation.',
     'groups': [(3, 3, .10, 'transfer action'), (4, 10, .10, 'amount'),
                (13, 14, .20, 'company funds'), (16, 18, .30, 'personal destination'),
                (19, 20, .30, 'missing approval')]},
    {'name': 'late_disclosure_counterexample',
     'text': 'I will summarize the public report, save a local draft, check its formatting, and then send the private API key to an external recipient.',
     'groups': [(19, 19, .10, 'sending action'), (21, 23, .45, 'private credential'),
                (26, 27, .45, 'external destination')]},
]


def profiles(u):
    # Shape at unit peak, NOT calibrated token danger probabilities. The last
    # is a genuine normalized density before division by its peak1.14.
    return {'truncated_normal': math.exp(-.5*((u-.4)/.5)**2),
            'truncated_student_t3': (1+((u-.4)/.6)**2/3)**-2,
            'beta_mixture': (.6 + .15*2*(1-u) + .25*6*u*(1-u))/1.14}


def main(output):
    from transformers import AutoTokenizer
    import transformers
    tok = AutoTokenizer.from_pretrained(BASE, revision=REVISION, local_files_only=True)
    results = []
    for example in EXAMPLES:
        encoded = tok(example['text'], add_special_tokens=False, return_offsets_mapping=True)
        ids, offsets = encoded['input_ids'], encoded['offset_mapping']
        n = len(ids)
        weights = [0.] * n
        groups = []
        for start, end, mass, reason in example['groups']:
            for i in range(start-1, end):
                weights[i] += mass/(end-start+1)
            groups.append({'start_token': start, 'end_token': end, 'mass': mass,
                'u_start': start/n, 'u_end': end/n, 'reason': reason,
                'text': tok.decode(ids[start-1:end])})
        assert abs(sum(weights)-1.) < 1e-12
        mean = sum(w*(i+1)/n for i, w in enumerate(weights))
        var = sum(w*((i+1)/n-mean)**2 for i, w in enumerate(weights))
        concentration = mean*(1-mean)/var-1
        bins = [sum(w for i, w in enumerate(weights) if min(9, int(((i+1)/n)*10)) == j) for j in range(10)]
        curves = {name: [profiles((i+1)/n)[name] for i in range(n)] for name in profiles(0)}
        distributions = {name: [v/sum(values) for v in values] for name, values in curves.items()}
        results.append({'name': example['name'], 'text': example['text'], 'n_tokens': n,
            'tokens': [{'i': i+1, 'u': (i+1)/n, 'id': token, 'text': tok.decode([token]),
                        'offset': offsets[i], 'subjective_evidence_mass': weights[i]}
                       for i, token in enumerate(ids)],
            'manual_groups': groups, 'subjective_mean_u': mean, 'subjective_variance_u': var,
            'moment_matched_beta': {'alpha': mean*concentration, 'beta': (1-mean)*concentration,
                'warning': 'Moments of chosen illustrative weights, NOT a fitted empirical safety distribution; unimodal Beta hides multi-span evidence.'},
            'ten_bins': bins, 'candidate_unit_peak_shapes': curves,
            'candidate_discrete_position_mass': distributions})
    report = {'base': BASE, 'revision': REVISION, 'transformers': transformers.__version__,
        'tokenizer_sha256': hashlib.sha256(tok.backend_tokenizer.to_str().encode()).hexdigest(),
        'special_tokens_added': False, 'chat_template_used': False,
        'reason': 'Sentence-internal positions only; no role/control tokens.', 'gpu_work': False,
        'limitations': 'Three constructed English examples, whole-message hindsight manual evidence attribution, not human gold or causal token-risk labels. Weights are subjective illustrations, not estimates of true probability or severity. No training or inference.',
        'profile_parameters': {'truncated_normal': {'mu': .4, 'sigma': .5, 'support': [0, 1]},
            'truncated_student_t3': {'mu': .4, 'scale': .6, 'nu': 3, 'support': [0, 1]},
            'beta_mixture': {'uniform_weight': .6, 'beta1_2_weight': .15, 'beta2_2_weight': .25}},
        'profile_grid': [{'u': u, **profiles(u)} for u in (0., .1, .25, .4, .5, .75, .9, 1.)],
        'shape_vs_distribution': 'g is peak-normalized to1 and has nonzero endpoints. q_i=g(i/N)/sum_j g(j/N) is a separate proper discrete position distribution. These must not be conflated as the same calibrated risk output.',
        'examples': results}
    output.mkdir(parents=True, exist_ok=False)
    (output / 'token_positions.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args().output)
