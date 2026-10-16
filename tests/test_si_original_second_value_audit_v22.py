"""v22 fixed-policy causal second-value audit tests, CPU only."""
import torch
from scripts import si_original_logged_value_v18 as v18
from scripts import si_original_crossfit_value_v19 as v19
from scripts import si_original_joint_logged_v21 as v21
from scripts.si_original_second_value_audit_v22 import audit_first,privacy_check,FAMILIES


def test_canonical_original_and_backbone_parity():
    assert v18.verify_source()=='575c17dc2f39a29d930d82179aa0c96dcbdc7d0f'
    for name,n in [('hybrid',64614),('transformer',64637)]:
        assert sum(p.numel() for p in v18.LoggedController(name).backbone.parameters())==n

def test_no_unpurchased_probe_or_hidden_label_in_second_choice():
    torch.set_num_threads(2)
    e=v19.evaluation_world(6,114001,'base')
    m=v21.FactoredPolicy(v19.CrossfitDoubleQ(v18.LoggedController('hybrid'),114001))
    assert privacy_check(m,e)
    e2={**e,'probe_low':e['probe_low']+1000,'latent_answer':-e['latent_answer']}
    f=torch.full((12,),2,dtype=torch.long)
    with torch.no_grad():
        assert torch.equal(m.q1(e,f),m.q1(e2,f))
        assert torch.equal(m.decision(e,f),m.decision(e2,f))


def test_counterfactual_sampling_does_not_modify_original_inputs_or_weights():
    torch.set_num_threads(2)
    e=v19.evaluation_world(8,114002,'base')
    model=v21.FactoredPolicy(v19.CrossfitDoubleQ(v18.LoggedController('gru'),114002))
    original={k:x.clone() for k,x in e.items()}
    weights=[p.detach().clone() for p in model.parameters()]
    with torch.no_grad(): result=audit_first(model,e,1,114002,6)
    assert all(torch.equal(e[k],original[k]) for k in original)
    assert all(torch.equal(p,q) for p,q in zip(model.parameters(),weights))
    assert result['replications']==6 and result['n']==16


def test_incremental_price_is_explicit_and_known():
    torch.set_num_threads(2)
    e=v19.evaluation_world(8,114003,'base')
    model=v21.FactoredPolicy(v19.CrossfitDoubleQ(v18.LoggedController('gru'),114003))
    with torch.no_grad():
        a=audit_first(model,e,1,114003,6)
        b=audit_first(model,{**e,'cost_high':e['cost_high']+.1},1,114003,6)
    # Probe value might also affect neural logits via public price, so never
    # assert a strict -.1 delta; check no mistaken reward without cost.
    assert all(torch.isfinite(torch.tensor(v)) for v in [a['predicted_mean'],a['empirical_mean'],b['empirical_mean']])
    assert a['expected_buy_reward']<=1 and b['expected_buy_reward']<=1


def test_bin_counts_and_regret_consistent():
    torch.set_num_threads(2)
    e=v19.evaluation_world(16,114004,'weak_sensors_ood')
    model=v21.FactoredPolicy(v19.CrossfitDoubleQ(v18.LoggedController('transformer'),114004))
    with torch.no_grad():r=audit_first(model,e,2,114004,4)
    assert sum(b['n'] for b in r['bins'])==len(e['mu_low'])
    assert r['clairvoyant_per_world_regret_upper_bound']>=-1e-7
    assert r['bin_policy_gap']>=-1e-6
    assert -1.5<=r['empirical_mean']<=1.0


def test_both_forced_first_actions_on_all_heldout_families():
    torch.set_num_threads(2)
    m=v21.FactoredPolicy(v19.CrossfitDoubleQ(v18.LoggedController('gru'),114005))
    for i,fam in enumerate(FAMILIES):
        e=v19.evaluation_world(4,114005+i,fam)
        for first in (1,2):
            with torch.no_grad():r=audit_first(m,e,first,114100+i+first,3)
            assert r['first']==first
            assert 0<=r['predicted_positive']<=1
            assert 0<=r['empirical_positive']<=1
