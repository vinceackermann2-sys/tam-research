import torch
from rlt_paper_merge import PaperGatedMerge

def test_no_feedback_is_exact_encoder_residual():
    merge=PaperGatedMerge(8,feedback_scale=0.)
    x=torch.randn(3,8);s=torch.randn(3,8)
    torch.testing.assert_close(merge(x,s),x,rtol=0,atol=0)

def test_published_equation_and_feedback_gradients():
    torch.manual_seed(817239)
    merge=PaperGatedMerge(8)
    x=torch.randn(3,8,requires_grad=True);s=torch.randn(3,8,requires_grad=True)
    r=merge.state_norm(s)
    expected=x+.1*torch.sigmoid(merge.gate(torch.cat((x,r),-1)))*merge.feedback(r)
    torch.testing.assert_close(merge(x,s),expected)
    merge(x,s).square().sum().backward()
    assert s.grad.norm()>0 and merge.gate.weight.grad.norm()>0 and merge.feedback.weight.grad.norm()>0
