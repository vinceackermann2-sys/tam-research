"""Current RLT paper equations9–11 only; not a complete paper reproduction.

Source: https://arxiv.org/html/2610.07591v1#S2.SS5
The repository's historical merge was a concatenation projection. This new
module preserves the encoder residual and gates normalized nonlinear feedback.
Other paper details (untied blocks, memory groups, positional scheme, data and
full training) still need independent implementation and validation.
"""
import torch
from torch import nn

class PaperGatedMerge(nn.Module):
    def __init__(self,width:int,feedback_scale:float=.1,eps:float=1e-6):
        super().__init__()
        if width<1 or feedback_scale<0:raise ValueError('invalid merge geometry')
        self.width=width;self.feedback_scale=feedback_scale
        self.state_norm=nn.RMSNorm(width,eps=eps)
        self.gate=nn.Linear(2*width,width,bias=True)
        self.feedback=nn.Linear(width,width,bias=False)
    def forward(self,encoder_state,previous_decoder_state):
        if encoder_state.shape!=previous_decoder_state.shape or encoder_state.shape[-1]!=self.width:
            raise ValueError('matching encoder/decoder state shapes required')
        r=self.state_norm(previous_decoder_state)
        g=torch.sigmoid(self.gate(torch.cat((encoder_state,r),dim=-1)))
        return encoder_state+self.feedback_scale*g*self.feedback(r)
