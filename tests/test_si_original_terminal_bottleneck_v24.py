"""V24 terminal-only ablation causal and identical-purchase integrity."""
import inspect
import torch
from scripts import si_original_logged_value_v18 as v18
from scripts import si_original_crossfit_value_v19 as v19
from scripts import si_original_uplift_calibration_v23 as v23
from scripts.si_original_terminal_bottleneck_v24 import (
    terminal_features, fit_terminal, TerminalPolicy, score_terminal_accuracy,
    _masks,
)


def test_full_canonical_original_source():
    assert v18.verify_source()=='575c17dc2f39a29d930d82179aa0c96dcbdc7d0f'
    assert sum(p.numel() for p in v18.LoggedController('hybrid').backbone.parameters())==64614
    assert sum(p.numel() for p in v18.LoggedController('transformer').backbone.parameters())==64637


def test_no_unpurchased_reading_or_private_labels_in_terminal_features():
    e=v19.evaluation_world(8,116001,'base')
    first=torch.ones(16,dtype=torch.long)
    modified={**e,'probe_high':e['probe_high']+999,'latent_answer':-e['latent_answer']}
    for typ in ('raw','likelihood'):
        assert torch.equal(terminal_features(e,first,feature=typ),terminal_features(modified,first,feature=typ))
        assert not torch.equal(terminal_features(e,first,torch.full_like(first,2),typ),terminal_features(modified,first,torch.full_like(first,2),typ))


def test_probe_absence_has_no_hidden_sensor_signal():
    e=v19.evaluation_world(8,116002,'base')
    zeros=torch.zeros(16,dtype=torch.long)
    for kind in ('raw','likelihood'):
        f=terminal_features(e,zeros,feature=kind)
        assert torch.equal(f[:,1:3],torch.zeros_like(f[:,1:3]))
        assert torch.equal(f[:,5:7],torch.zeros_like(f[:,5:7]))


def test_factual_logged_terminal_feedback_training_and_finite_weights():
    torch.set_num_threads(2)
    for kind in ('raw','likelihood'):
        w,s=fit_terminal(116003,n_train=400,kind=kind)
        assert s['n_train']==400
        assert len(w)==(7 if kind=='raw' else 9)
        assert torch.isfinite(w).all()
    source=inspect.getsource(fit_terminal)
    assert 'logged_feedback(' in source
    assert 'teacher_first(' not in source
    assert 'latent_answer' not in source


def test_exact_known_gaussian_decision_from_only_purchased_data():
    e=v19.evaluation_world(8,116004,'base')
    dummy=object()
    method=TerminalPolicy(dummy,'bayes_terminal')
    f=torch.ones(16,dtype=torch.long)
    pred=method.decision(e,f)
    assert torch.allclose(pred,e['mu_low']*e['probe_low'])
    pred2=method.decision(e,f,3-f)
    assert torch.allclose(pred2,e['mu_low']*e['probe_low']+e['mu_high']*e['probe_high'])


def test_same_frozen_purchase_across_all_final_classifiers():
    torch.set_num_threads(2)
    model=v23.CorrectedPolicy(v23.v21.FactoredPolicy(v19.CrossfitDoubleQ(v18.LoggedController('gru'),116005)),
         {'weights':[torch.zeros(14),torch.zeros(14)],'shifts':{1:0,2:0}},'ridge')
    e=v19.evaluation_world(16,116005,'base')
    weights=torch.zeros(7)
    methods=[TerminalPolicy(model,m,weights if m=='raw' else None) for m in ('original','raw','bayes_terminal')]
    with torch.no_grad():r=[score_terminal_accuracy(e,m) for m in methods]
    for key in ('first_low','first_high','second_buy','no_probe','mean_cost'):
        assert all(abs(z[key]-r[0][key])<1e-8 for z in r)


def test_terminal_ridge_does_not_mutate_original_weights():
    torch.set_num_threads(2)
    base=v18.LoggedController('hybrid')
    state=[p.detach().clone() for p in base.parameters()]
    fit_terminal(116006,n_train=400,kind='likelihood')
    assert all(torch.equal(p,s) for p,s in zip(base.parameters(),state))


def test_forced_terminal_probe_costs_are_exact_and_never_hidden():
    from scripts.si_original_terminal_bottleneck_v24 import score_forced_probe
    e=v19.evaluation_world(8,116007,'base')
    p=TerminalPolicy(object(),'bayes_terminal')
    with torch.no_grad():
        cheap=score_forced_probe(e,p,'cheap')
        strong=score_forced_probe(e,p,'strong')
        both=score_forced_probe(e,p,'both')
    assert abs(cheap['mean_cost']-float(e['cost_low'].mean()))<1e-6
    assert abs(strong['mean_cost']-float(e['cost_high'].mean()))<1e-6
    assert abs(both['mean_cost']-float((e['cost_low']+e['cost_high']).mean()))<1e-6


def test_cached_evaluation_identical_to_independent_purchase_paths():
    from scripts.si_original_terminal_bottleneck_v24 import score_cached_terminal
    torch.set_num_threads(2)
    e=v19.evaluation_world(12,116008,'base')
    m=v23.CorrectedPolicy(v23.v21.FactoredPolicy(v19.CrossfitDoubleQ(v18.LoggedController('hybrid'),116008)),
         {'weights':[torch.zeros(14),torch.zeros(14)],'shifts':{1:0,2:0}},'ridge')
    methods={'original':TerminalPolicy(m,'original'),
             'bayes_terminal':TerminalPolicy(m,'bayes_terminal'),
             'raw':TerminalPolicy(m,'raw',torch.zeros(7))}
    fast=score_cached_terminal(e,methods)
    for k,v in methods.items():
        slow=score_terminal_accuracy(e,v)
        for key in fast[k]:
            if slow[key] is None: assert fast[k][key] is None
            else:assert abs(fast[k][key]-slow[key])<1e-7,(k,key)
