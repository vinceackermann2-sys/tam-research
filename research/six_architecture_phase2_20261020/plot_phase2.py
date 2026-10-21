import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path(__file__).parent
plt.rcParams.update({'figure.dpi':150,'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
def main():
    if (ROOT/'EVIDENCE.json').exists():
        e=json.loads((ROOT/'EVIDENCE.json').read_text());rows=list(e['results'].values())
        if rows:
            stages=sorted({r['steps'] for r in rows});first=[r for r in rows if r['steps']==stages[0]]
            fig,axes=plt.subplots(1,2,figsize=(10,3.8))
            for r in first:
                axes[0].scatter(r['train_tokens_per_second']/1000,r['validation']['nll'],s=65)
                axes[0].annotate(r['variant'],(r['train_tokens_per_second']/1000,r['validation']['nll']),xytext=(5,5),textcoords='offset points')
            axes[0].set(xlabel='Training throughput (thousand tokens/s)',ylabel='Development NLL (lower is better)',title=f"Matched {first[0]['tokens_seen']/1e6:.0f}M-token stage")
            for arm in sorted({r['variant'] for r in rows}):
                rs=sorted([r for r in rows if r['variant']==arm],key=lambda z:z['tokens_seen'])
                axes[1].plot([r['tokens_seen']/1e6 for r in rs],[r['validation']['nll'] for r in rs],marker='o',label=arm)
            axes[1].set(xlabel='Cumulative optimizer exposures (million)',ylabel='Development NLL',title='Frozen 2B-horizon schedule prefixes');axes[1].legend(fontsize=8)
            fig.suptitle('Near-100M language models · A10040GB · one seed · context512');fig.tight_layout();fig.savefig(ROOT/'language_panel.png');plt.close(fig)
    rs=[json.loads(p.read_text()) for p in sorted((ROOT/'results/si_repair').glob('*.RESULT.json'))]
    if rs:
        x=np.arange(len(rs));fig,ax=plt.subplots(figsize=(7,4.2));width=.25
        ax.bar(x-width,[r['baseline']['accuracy']['native_gated'] for r in rs],width,label='Before head training')
        ax.bar(x,[r['repaired']['accuracy']['native_gated'] for r in rs],width,label='After factual head training')
        ax.bar(x+width,[r['repaired']['accuracy']['direct'] for r in rs],width,label='Unchanged direct policy')
        ax.set(xticks=x,xticklabels=[str(r['seed']) for r in rs],ylim=(0,1.12),ylabel='Action accuracy',title='SI search-head repair · small finite one-step cue task')
        ax.legend(fontsize=7,loc='upper center',bbox_to_anchor=(.5,-.15),ncol=3)
        fig.text(.5,.015,'Repaired gate selects direct action; forced search also scores100%.',ha='center',fontsize=8)
        fig.tight_layout(rect=(0,.06,1,1));fig.savefig(ROOT/'si_head_repair.png');plt.close(fig)
    rs=[json.loads(p.read_text()) for p in sorted((ROOT/'results/rlt_gate').glob('*.RESULT.json'))]
    if rs:
        fig,ax=plt.subplots(figsize=(7,3.5))
        for r in rs:
            zs=[z for z in r['rows'] if z['carry']]
            ax.plot([z['length']-1 for z in zs],[z['accuracy'] for z in zs],marker='o',linestyle='-' if r['arm']=='native' else '--',label=f"{r['seed']} {r['arm']}")
        ax.axhline(.5,color='gray',lw=1,label='Erased-carry control');ax.set(xlabel='Distance from stored SET to query',ylabel='Paired-prefix accuracy',ylim=(0,1.05),title='Bounded RLT retention · fresh seeds · input-type gate prior');ax.legend(fontsize=7,ncol=2)
        fig.tight_layout();fig.savefig(ROOT/'rlt_gate_repair.png');plt.close(fig)
if __name__=='__main__':main()
