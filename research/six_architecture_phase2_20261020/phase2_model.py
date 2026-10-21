"""Prospective larger Conv7 comparison; same block equations as phase-one source.

Width 512/head dim 64 supports the A100 flash-attention kernel. Full-attention
FFN width is reduced to 1968 to obey the strict actual 100M parameter ceiling.
Removed attention capacity is compensated with active FFN parameters.
"""
import torch
from torch import nn
from torch.nn import functional as F

VOCAB=50257
class Attention(nn.Module):
    def __init__(self,d=512):
        super().__init__(); self.heads=8
        self.qkv=nn.Linear(d,3*d,bias=False); self.out=nn.Linear(d,d,bias=False)
    def forward(self,x):
        b,t,d=x.shape
        q,k,v=[z.view(b,t,self.heads,d//self.heads).transpose(1,2) for z in self.qkv(x).chunk(3,-1)]
        return self.out(F.scaled_dot_product_attention(q,k,v,is_causal=True).transpose(1,2).contiguous().view(b,t,d))

class Block(nn.Module):
    def __init__(self,d,ff,attention,conv):
        super().__init__()
        self.norm_attention=nn.LayerNorm(d) if attention else None
        self.attention=Attention(d) if attention else None
        self.norm_ff=nn.LayerNorm(d)
        self.feedforward=nn.Sequential(nn.Linear(d,ff,bias=False),nn.GELU(),nn.Linear(ff,d,bias=False))
        self.conv=nn.Conv1d(d,d,7,groups=d,bias=False) if conv else None
        if conv:self.conv_gain=nn.Parameter(torch.zeros(()))
    def forward(self,x):
        if self.attention is not None:x=x+self.attention(self.norm_attention(x))
        if self.conv is not None:x=x+self.conv_gain*self.conv(F.pad(self.norm_ff(x).transpose(1,2),(6,0))).transpose(1,2)
        return x+self.feedforward(self.norm_ff(x))

class LanguageModel(nn.Module):
    def __init__(self,variant,d=512,layers=24,base_ff=1968,positions=1024):
        super().__init__();self.d=d;self.variant=variant;self.positions=positions
        if variant not in ('transformer','attention8','conv7'):raise ValueError(variant)
        attention_count=layers if variant=='transformer' else layers//3
        saved=(layers-attention_count)*(4*d*d+2*d)
        added=(layers-attention_count)*(7*d+1) if variant=='conv7' else 0
        self.ff=base_ff+round((saved-added)/(layers*2*d))
        self.token_emb=nn.Embedding(VOCAB,d);self.pos_emb=nn.Embedding(positions,d)
        self.blocks=nn.ModuleList([Block(d,self.ff,variant=='transformer' or (i+1)%3==0,
            variant=='conv7' and (i+1)%3!=0) for i in range(layers)])
        self.norm=nn.LayerNorm(d)
        self.apply(self.init_weights)
    @staticmethod
    def init_weights(m):
        if isinstance(m,(nn.Linear,nn.Embedding)):
            nn.init.normal_(m.weight,mean=0,std=.02)
            if isinstance(m,nn.Linear) and m.bias is not None:nn.init.zeros_(m.bias)
    def hidden(self,tokens):
        if tokens.size(1)>self.positions:raise ValueError('context cap')
        x=self.token_emb(tokens)+self.pos_emb(torch.arange(tokens.size(1),device=tokens.device))[None]
        for block in self.blocks:x=block(x)
        return self.norm(x)
    def forward(self,tokens):return F.linear(self.hidden(tokens),self.token_emb.weight)
    def loss(self,tokens,targets):return F.cross_entropy(self(tokens).float().flatten(0,1),targets.flatten())

def count(model):return sum(p.numel() for p in model.parameters())
