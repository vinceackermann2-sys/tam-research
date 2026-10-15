import torch
from scripts.si_original_causal_transition_v1 import batch_worlds


def test_worlds_reproducible():
    a=batch_worlds(4,16,94001)
    b=batch_worlds(4,16,94001)
    assert all(torch.equal(x,y) for x,y in zip(a,b))


def test_reversal_changes_future_without_retrocausality():
    stable=batch_worlds(4,16,94001,"stable")
    switched=batch_worlds(4,16,94001,"switch")
    for a,b in zip(stable,switched):
        assert torch.equal(a[:,:8],b[:,:8])
    assert not torch.equal(stable[2][:,8:],switched[2][:,8:])


def test_no_hidden_rule_in_model_inputs():
    x,actions,target=batch_worlds(4,16,94001)
    assert x.shape[-1]==3 and actions.shape[-1]==4
    assert actions.sum(-1).eq(1).all()
    assert target.shape==x.shape
