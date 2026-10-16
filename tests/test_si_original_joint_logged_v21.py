"""v21 causal action selection, true logged arm and source checks."""
import inspect
import torch
import torch.nn.functional as F
from scripts import si_original_logged_value_v18 as v18
from scripts import si_original_crossfit_value_v19 as v19
from scripts import si_original_direct_first_value_v20 as v20
from scripts.si_original_active_experiment_v17 import world
from scripts.si_original_joint_logged_v21 import FactoredPolicy, fit_second, fit_first, second_calibration


def test_exact_canonical_original_and_matched_models():
    assert v18.verify_source() == '575c17dc2f39a29d930d82179aa0c96dcbdc7d0f'
    assert sum(p.numel() for p in v18.LoggedController('hybrid').backbone.parameters())==64614
    assert sum(p.numel() for p in v18.LoggedController('transformer').backbone.parameters())==64637


def test_unbought_probe_and_truth_invisible_to_first_and_second_policy():
    torch.set_num_threads(2)
    e=world(6,113001)
    altered={**e,'probe_high':e['probe_high']+1000,'latent_answer':-e['latent_answer']}
    net=FactoredPolicy(v19.CrossfitDoubleQ(v18.LoggedController('hybrid'),113001))
    first=torch.ones(12,dtype=torch.long)
    with torch.no_grad():
        assert torch.equal(net.q0(e),net.q0(altered))
        assert torch.equal(net.q1(e,first),net.q1(altered,first))
        assert torch.equal(net.decision(e,first),net.decision(altered,first))


def test_factual_second_success_update_has_zero_unchosen_gradient():
    torch.set_num_threads(2)
    e=world(5,113002)
    net=FactoredPolicy(v19.CrossfitDoubleQ(v18.LoggedController('gru'),113002))
    f=torch.ones(10,dtype=torch.long)
    qa,qb=net.second_success_logits(e,f)
    chosen=torch.ones(10,dtype=torch.long)
    F.binary_cross_entropy_with_logits(qa.gather(1,chosen[:,None]).squeeze(1),torch.ones(10)).backward()
    assert torch.equal(net.success_a.weight.grad[0],torch.zeros_like(net.success_a.weight.grad[0]))
    assert net.success_a.weight.grad[1].abs().sum()>0
    assert net.success_b.weight.grad is None


def test_cost_change_is_exact_arithmetic_not_model_guess():
    e=world(5,113003)
    net=FactoredPolicy(v19.CrossfitDoubleQ(v18.LoggedController('gru'),113003))
    first=torch.ones(10,dtype=torch.long)
    altered={**e,'cost_high':e['cost_high']+.2}
    # Neural backbone sees prices, so predictions can change with price; this
    # check isolates the explicit subtraction from predicted success logits.
    with torch.no_grad():
        a=net.q1(e,first);b=net.q1(altered,first)
        qa,qb=net.second_success_logits(e,first)
        ra,rb=net.second_success_logits(altered,first)
        predicted_difference=(ra.sigmoid()+rb.sigmoid()-qa.sigmoid()-qb.sigmoid())*.5
        assert torch.allclose((b-a)-predicted_difference,torch.tensor([0.,-.2]),atol=1e-6)


def test_first_logged_acceptance_independent_of_hidden_truth_and_frozen_actor():
    e=world(6,113004)
    net=FactoredPolicy(v19.CrossfitDoubleQ(v18.LoggedController('gru'),113004))
    first=torch.ones(12,dtype=torch.long)
    second=v18.random_boolean(12,113004)
    stop=v18.random_boolean(12,113005)
    with torch.no_grad():
        _,accepted=v20.logged_first_trajectories(e,net,first,second,stop)
        altered={**e,'latent_answer':-e['latent_answer']}
        _,accepted_other=v20.logged_first_trajectories(altered,net,first,second,stop)
    assert torch.equal(accepted,accepted_other)


def test_no_bayesian_teacher_labels_in_optimizer():
    source=inspect.getsource(fit_second)+inspect.getsource(fit_first)
    assert 'teacher_first(' not in source and 'second_purchase(' not in source
    assert 'logged_feedback(' in source or 'logged_first_trajectories(' in source
    assert 'cross_entropy_with_logits(' in source


def test_small_cpu_training_finite_and_original_weights_frozen():
    torch.set_num_threads(2)
    base,_=v19.train_one(113005,'gru',2,2,8)
    model=FactoredPolicy(base)
    before={k:p.detach().clone() for k,p in model.core.named_parameters()}
    fit_second(113005,model,steps=2,batch=8)
    fit_first(113005,model,steps=2,batch=8)
    for k,p in model.core.named_parameters(): assert torch.equal(before[k],p)
    e=world(4,113006)
    q=second_calibration(model,e,113006)
    assert 0<=q<10
    assert all(torch.isfinite(x).all() for x in model.parameters())


def test_contextual_ridge_trained_only_on_observed_actions_and_public_features():
    from scripts import si_original_linear_context_v21 as linear
    torch.set_num_threads(2)
    model,stats=linear.train(113007,n_train=600)
    assert stats['q0_total']==600
    assert 100 < stats['q0_accepted'] < 550
    assert all(torch.isfinite(w).all() for w in model.q0_weights+model.q1_weights)
    ep=world(12,113007)
    altered={**ep,'latent_answer':-ep['latent_answer'],'probe_high':ep['probe_high']+1000}
    # Before a probe no label or future probe enters the observed features.
    assert torch.equal(linear.q0_features(ep),linear.q0_features(altered))
    f=torch.ones(24,dtype=torch.long)
    assert torch.equal(linear.q1_features(ep,f),linear.q1_features(altered,f))

def test_contextual_ridge_cost_responsiveness_and_finite_policy():
    from scripts import si_original_linear_context_v21 as linear
    m,stats=linear.train(113008,n_train=1000)
    a=world(10,113008)
    b={**a,'cost_high':a['cost_high']+.13}
    f=torch.ones(20,dtype=torch.long)
    assert torch.allclose(m.q1(b,f)-m.q1(a,f),torch.tensor([0.,-.13]),atol=1e-6)
    from scripts.si_original_crossfit_value_v19 import evaluation_world
    r=linear.score(m,evaluation_world(10,113008,'strong_discount_ood'))
    assert -1<=r['reward']<=1
    assert 0<=r['first_high']<=1
