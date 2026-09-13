import torch
from tam_research.physics_tokenizer_centroid_dynamics import CentroidResidualDynamics, norm_to_tokens, tokens_to_norm

def test_centroid_token_roundtrip_exact_for_codebook_values():
    cb=torch.tensor([[-1.,-1.],[-1.,1.],[1.,-1.],[1.,1.]])
    ids=torch.tensor([[[0,1,2,3]]])
    states=tokens_to_norm(ids,cb,side=2)[:,0]
    assert torch.equal(norm_to_tokens(states,cb),ids[:,0])

def test_residual_model_zero_init_is_token_state_persistence():
    model=CentroidResidualDynamics(context=3,width=8); x=torch.randn(2,3,2,4,4)
    with torch.no_grad(): y=model(x)
    assert torch.allclose(y,x[:,-1])
