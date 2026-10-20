import os,subprocess,sys
from pathlib import Path
import torch
from phase2_aligned_v2 import make,count
from phase2_model import Block

torch.set_num_threads(2)

def test_actual_parameter_counts_and_integer_exposure_caps():
    counts={}
    for arm in ('transformer','attention8','conv7','cpw_local'):
        with torch.device('meta'):m=make(arm)
        counts[arm]=count(m)
    assert counts=={'transformer':99837440,'attention8':99558912,'conv7':99616272,'cpw_local':99674432}
    assert max(counts.values())<=100_000_000
    assert (2_000_000_000//8192)*8192==1_999_994_880

def test_active_convolution_is_causal_and_receives_gradients():
    torch.manual_seed(2);b=Block(16,32,False,True);b.conv_gain.data.fill_(.3)
    x=torch.randn(2,12,16);changed=x.clone();changed[:,6:]=torch.randn(2,6,16)
    assert torch.equal(b(x)[:,:6],b(changed)[:,:6])
    b(x).square().mean().backward()
    assert b.conv.weight.grad.norm()>0

def test_cpw_scaled_equations_match_native_source():
    repo=Path(os.environ.get('CPW_REPO',str(Path(__file__).parents[2]/'work/cpw')))
    assert repo.is_dir(),'Set CPW_REPO to the canonical pinned CPW checkout'
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()=='aad622d1eea591545a3482618ca8b269e4d9d43f'
    sys.path.insert(0,str(repo))
    from tam_research.cpw_v1.model import CPWV1Config
    from tam_research.cpw_v3_sparse_world.model import SparseWorldCPWResearchLM
    torch.manual_seed(91);derived=make('cpw_local').eval()
    native=SparseWorldCPWResearchLM('sequence_only',CPWV1Config(vocab_size=50257,d_model=704,n_layers=15,max_seq_len=1024,ff_mult=4,sequence_predictor_rank=192)).eval()
    state=derived.state_dict();mapped={}
    for name,p in native.state_dict().items():
        key=name.replace('.mixer.sequence.down.','.down.').replace('.mixer.sequence.up.','.up.')
        if key=='lm_head.weight':key='token_emb.weight'
        mapped[name]=state[key]
    native.load_state_dict(mapped)
    x=torch.randint(50257,(1,20));changed=x.clone();changed[:,12:]=torch.randint(50257,(1,8))
    with torch.no_grad():
        expected=native(x);actual=derived(x)
        assert torch.equal(expected,actual)
        assert torch.equal(actual[:,:12],derived(changed)[:,:12])
    assert count(native)==count(derived)
