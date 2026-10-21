"""Plot matched scientific endpoints without mixing A100 or backup midpoints."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path(__file__).parent


def main():
    summary=json.loads((ROOT/'L4_SUMMARY.json').read_text())
    comparisons=summary['scientific_comparisons']
    labels={'attention8':'Eight-attention parent','conv7':'Conv7','cpw_local':'CPW local'}
    colors={'attention8':'#778899','conv7':'#2676b8','cpw_local':'#c88331'}
    plt.rcParams.update({'figure.dpi':150,'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    for tokens in sorted({r['tokens_seen'] for r in comparisons}):
        rows=[r for r in comparisons if r['tokens_seen']==tokens];y=np.arange(len(rows))
        fig,axes=plt.subplots(1,2,figsize=(10.4,4.1),sharey=True)
        delta=np.array([r['delta_mean_nll'] for r in rows])
        intervals=np.array([r['approximate_95_percent_block_bootstrap_interval'] for r in rows])
        axes[0].barh(y,delta,color=[colors[r['variant']] for r in rows],height=.55)
        axes[0].errorbar(delta,y,xerr=np.stack([delta-intervals[:,0],intervals[:,1]-delta]),fmt='none',ecolor='#222222',capsize=3)
        axes[0].axvline(0,color='#555555',lw=1)
        axes[0].set(yticks=y,yticklabels=[labels[r['variant']] for r in rows],xlabel='Development NLL minus Transformer',title='Quality: lower is better')
        for i,value in enumerate(delta):
            axes[0].annotate(f'{value:+.4f}',(value,i),xytext=(7 if value>=0 else -7,-19),textcoords='offset points',ha='left' if value>=0 else 'right',fontsize=9)
        speed=[r['training_speed_ratio'] for r in rows]
        axes[1].barh(y,speed,color=[colors[r['variant']] for r in rows],height=.55)
        axes[1].axvline(1,color='#555555',lw=1,ls='--')
        axes[1].set(xlabel='Timed training throughput / Transformer',title='Speed: higher is better',xlim=(0,max(speed)*1.2))
        for i,value in enumerate(speed):axes[1].text(value+.015,i,f'{value:.4f}x',va='center',fontsize=9)
        fig.suptitle(f'Near-100M parameters | {tokens:,} training exposures | L4 | seed {summary["seed"]}')
        fig.text(.5,.025,'Quality intervals: descriptive token-block bootstrap, one paired seed. Timed throughput excludes input/evaluation/backup.',ha='center',fontsize=8)
        fig.tight_layout(rect=(0,.08,1,.94))
        fig.savefig(ROOT/f'l4_language_gate_{round(tokens/1e6)}M.png')
        plt.close(fig)


if __name__=='__main__':main()
