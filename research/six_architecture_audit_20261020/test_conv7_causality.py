"""Future isolation must hold when the convolution gates are active."""
import torch
from conv7_language import ScheduledLM

def test_nonzero_convolution_gates_cannot_read_future_tokens():
    torch.set_num_threads(2);torch.manual_seed(410238)
    model=ScheduledLM(16,'conv7').eval()
    for block in model.blocks:
        if block.conv is not None:block.conv_gain.data.fill_(.5)
    original=torch.randint(0,50257,(2,12));changed=original.clone()
    changed[:,6:]=(changed[:,6:]+123)%50257
    with torch.no_grad():
        first=model(original);second=model(changed)
    torch.testing.assert_close(first[:,:6],second[:,:6],rtol=0,atol=1e-6)
    assert not torch.allclose(first[:,6:],second[:,6:])
