"""Plot existing hypothetical priors and subjective Qwen-token evidence examples."""
import argparse
import json
import math
from pathlib import Path
import textwrap

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

COLORS = ['#2563eb', '#dc2626', '#16a34a']
NAMES = ['Normal: μ=0.4, σ=0.5', 'Student-t: μ=0.4, s=0.6, ν=3',
         'Смесь: 0.60 Uniform + 0.15 Beta(1,2) + 0.25 Beta(2,2)']
KEYS = ['truncated_normal', 'truncated_student_t3', 'beta_mixture']


def curves(u):
    return [math.exp(-.5*((u-.4)/.5)**2), (1+((u-.4)/.6)**2/3)**-2,
            (.6+.3*(1-u)+1.5*u*(1-u))/1.14]


def main(root):
    data = json.loads((root / 'token_positions.json').read_text())
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11})
    x = [i/1000 for i in range(1001)]
    ys = [[curves(u)[k] for u in x] for k in range(3)]
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), layout='constrained')
    for k, (name, color, y) in enumerate(zip(NAMES, COLORS, ys)):
        area = sum((y[i]+y[i+1])*.0005 for i in range(1000))
        axes[0].plot(x, y, label=name, color=color, lw=2.5, linestyle=['-', '--', ':'][k])
        axes[1].plot(x, [v/area for v in y], color=color, lw=2.5, linestyle=['-', '--', ':'][k])
        axes[0].scatter([0, 1], [y[0], y[-1]], color=color, s=36, zorder=4)
    for ax in axes:
        ax.set(xlim=(0, 1), xlabel='Нормализованная позиция u = i / N')
        ax.axvline(.4, color='#64748b', alpha=.5, ls='--')
        ax.grid(alpha=.2)
    axes[0].set(title='Профиль g(u): максимум = 1', ylabel='Высота формы — не измеренная вероятность', ylim=(0, 1.07))
    axes[1].set(title='Плотность f(u): площадь = 1', ylabel='Плотность позиции (может быть больше 1)', ylim=(0, 1.5))
    axes[0].legend(loc='lower center', fontsize=9)
    fig.suptitle('Кандидатные prior: высокий старт, широкий пик, ненулевой конец', fontsize=15)
    fig.supxlabel('Это искусственные формы, не выходы модели и не установленное распределение опасности.', fontsize=11)
    fig.savefig(root / 'prior_profiles.png', dpi=160, facecolor='white')
    fig.savefig(root / 'prior_profiles.svg', facecolor='white')
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(15, 14), layout='constrained')
    for index, (ax, example) in enumerate(zip(axes, data['examples']), 1):
        positions = [t['u'] for t in example['tokens']]
        manual = [t['subjective_evidence_mass'] for t in example['tokens']]
        n = example['n_tokens']
        ax.bar(positions, manual, width=.7/n, color='#64748b', alpha=.65,
               label='Ручные веса свидетельств — условная иллюстрация, не gold')
        for name, color, key in zip(NAMES, COLORS, KEYS):
            ax.plot(positions, example['candidate_discrete_position_mass'][key], color=color,
                    lw=2, label=name.split(':')[0])
        for group in example['manual_groups']:
            center = (group['u_start']+group['u_end'])/2
            height = group['mass']/(group['end_token']-group['start_token']+1)
            ax.annotate(f'{group["start_token"]}–{group["end_token"]}: {group["text"].strip()}',
                xy=(center, height), xytext=(0, 9), textcoords='offset points',
                ha='center', fontsize=8, rotation=12)
        ax.set(xlim=(0, 1.03), ylim=(0, max(manual)*1.5),
            xlabel='u = номер Qwen-токена / число токенов', ylabel='Масса qᵢ (сумма по токенам = 1)',
            title=f'Пример {index}, N={n}\n'+textwrap.fill(example['text'], width=115))
        ax.grid(alpha=.2)
    axes[0].legend(loc='upper left', fontsize=8)
    fig.suptitle('Где находятся свидетельства в трёх сконструированных предложениях', fontsize=15)
    fig.supxlabel('Qwen3.5-4B-Base: реальные token IDs. Столбцы — субъективная ретроспективная разметка автора,\n'
                   'линии — заранее выбранные prior. Не измерение риска префикса; по этим примерам нельзя выбрать лучший prior.', fontsize=11)
    fig.savefig(root / 'manual_token_examples.png', dpi=150, facecolor='white')
    fig.savefig(root / 'manual_token_examples.svg', facecolor='white')
    plt.close(fig)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    main(p.parse_args().root)
