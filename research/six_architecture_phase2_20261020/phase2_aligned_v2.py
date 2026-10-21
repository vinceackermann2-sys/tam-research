"""Floor FFN width to16: nearest rounding could exceed100M and was rejected.

The rejected Conv7 configuration was never trained. This is the selected
prospective model source; earlier hardware trials are retained separately.
"""
from torch import nn
from phase2_model import LanguageModel,count,VOCAB
from phase2_aligned_model import CPWLocalLM

class AlignedLM(LanguageModel):
    def __init__(self,variant):
        super().__init__(variant)
        ff=16*(self.ff//16)
        for b in self.blocks:
            b.feedforward=nn.Sequential(nn.Linear(512,ff,bias=False),nn.GELU(),nn.Linear(ff,512,bias=False))
        self.ff=ff;self.apply(self.init_weights)

def make(variant):return CPWLocalLM() if variant=='cpw_local' else AlignedLM(variant)
