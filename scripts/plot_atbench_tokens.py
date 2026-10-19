"""Plot saved true token readouts; never invoke a model or smooth the data."""
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

source,output=map(Path,sys.argv[1:3])
data=json.loads(source.read_text())
colors=['#2563eb','#16a34a','#ea580c']
names=['E1: extract_news_article_text','E2: create_event','E3: Complete']
fig,axes=plt.subplots(2,1,figsize=(14,8),sharex=True,layout='constrained')
for ax,maximum,title in zip(axes,[1,.25],['Full scale: 0–1','Zoom: 0–0.25 (same scores)']):
    offset=0
    for event,color,label in zip(data['events'],colors,names):
        scores=event['action_scores'];x=list(range(offset+1,offset+len(scores)+1))
        if offset:ax.axvline(offset+.5,color='#94a3b8',linestyle=':',linewidth=1)
        ax.plot(x,scores,color=color,label=label,linewidth=1.7)
        ax.scatter(x[-1],scores[-1],color=color,s=35,zorder=4)
        if maximum<.5:
            ax.annotate(f"{scores[-1]:.4f}",(x[-1],scores[-1]),xytext=(-15,12),
                        textcoords='offset points',color=color,fontweight='bold')
        offset+=len(scores)
    if maximum>=.5:ax.axhline(.5,color='#dc2626',linestyle='--',label='Decision threshold 0.5')
    ax.set(ylim=(0,maximum),ylabel='Classifier score',title=title)
    ax.grid(alpha=.2)
axes[0].legend(loc='upper right',fontsize=9)
axes[1].set_xlabel('Token index across proposed actions (43 + 147 + 75 tokens)')
model_name='Mamba + LoRA' if data['model']=='lora_validation_epoch2' else data['model']
endpoint_max=max(e['action_scores'][-1] for e in data['events'])
fig.suptitle(f"ATBench #1 · {model_name} (epoch {data['selected']['epoch']})\n"
    'Separate full-prefix replay for each action; only proposed-action tokens shown.\n'
    f'Gold: unsafe trajectory | Max action-endpoint score: {endpoint_max:.4f}',fontsize=13)
fig.supxlabel('Dots = endpoints used for trajectory scoring; internal token peaks are NOT aggregated.',fontsize=10)
fig.savefig(output,dpi=160,facecolor='white')
