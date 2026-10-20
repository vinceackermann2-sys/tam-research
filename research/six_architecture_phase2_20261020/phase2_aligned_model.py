"""Hardware-aligned derivative frozen before any language training.

All FFN widths are rounded to nearest16 before initialization, with <=0.5%
actual parameter imbalance. CPW is the native sequence-only branch equations,
width/rank enlarged; it is explicitly not the unvalidated long-memory AFM.
"""
import torch
from torch import nn
from torch.nn import functional as F
from phase2_model import LanguageModel,Attention,Block,VOCAB,count

class AlignedLM(LanguageModel):
    def __init__(self,variant):
        super().__init__(variant)
        ff=16*round(self.ff/16)
        for b in self.blocks:
            b.feedforward=nn.Sequential(nn.Linear(512,ff,bias=False),nn.GELU(),nn.Linear(ff,512,bias=False))
        self.ff=ff
        self.apply(self.init_weights)

class CPWBlock(nn.Module):
    def __init__(self):
        super().__init__();self.norm1=nn.LayerNorm(704);self.norm2=nn.LayerNorm(704)
        self.down=nn.Linear(704,192,bias=False);self.up=nn.Linear(192,704,bias=False)
        self.ff=nn.Sequential(nn.Linear(704,2816,bias=False),nn.GELU(),nn.Linear(2816,704,bias=False))
    def forward(self,x):
        z=self.norm1(x);previous=torch.cat((torch.zeros_like(z[:,:1]),z[:,:-1]),dim=1)
        x=x+self.up(F.gelu(self.down(previous)))
        return x+self.ff(self.norm2(x))

class CPWLocalLM(nn.Module):
    def __init__(self):
        super().__init__();self.variant='cpw_local';self.d=704;self.ff=2816;self.positions=1024
        self.token_emb=nn.Embedding(VOCAB,704);self.pos_emb=nn.Embedding(1024,704)
        self.blocks=nn.ModuleList([CPWBlock() for _ in range(15)]);self.norm=nn.LayerNorm(704)
        self.apply(LanguageModel.init_weights)
    hidden=LanguageModel.hidden
    forward=LanguageModel.forward
    loss=LanguageModel.loss

def make(variant):return CPWLocalLM() if variant=='cpw_local' else AlignedLM(variant)
