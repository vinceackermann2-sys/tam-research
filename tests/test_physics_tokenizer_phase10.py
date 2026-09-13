import numpy as np

from tam_research import physics_tokenizer_phase10 as p10


def test_exact_18bit_packing_and_endpoints():
    inv=p10.validate_invariants()
    assert inv['packed_min']==0 and inv['packed_max']==262143 and inv['nominal_bits']==18
    assert np.allclose(inv['endpoint_z'],[-16.0,16.0])


def test_kdtree_assignment_matches_bruteforce():
    rng=np.random.default_rng(8675309)
    x=rng.normal(size=(1500,2)).astype(np.float32); c=rng.normal(size=(97,2)).astype(np.float32)
    kd=p10.nearest(x,c); brute=((x[:,None,:]-c[None,:,:])**2).sum(-1).argmin(1)
    np.testing.assert_array_equal(kd,brute)


def test_companded_prefix_causal_and_static_tolerance():
    m=np.zeros(2,np.float32); s=np.ones(2,np.float32)
    dummy=(m,s,np.array([[0.,0.]],np.float32),np.array([[0.,0.]],np.float32))
    raw=np.zeros((2,4,2,3,3),np.float32); changed=raw.copy(); changed[:,3]=11.
    a=p10.encode_comp_seq(raw,dummy,m,s); b=p10.encode_comp_seq(changed,dummy,m,s)
    np.testing.assert_array_equal(a[:,:3],b[:,:3])
    assert float(np.max(np.abs(a))) < .05


def test_encoders_do_not_mutate_shared_raw_trajectory():
    raw,_=p10.gen_wave(3,5,8,(.6,1.0),424242,3); before=raw.copy()
    sm,ss=p10.stats(raw); points=p10.flat2(p10.norm5(raw,sm,ss)); c1,c2=p10.fit_rvq(points,16,4,101,102,iters=2,max_points=1000); state=(sm,ss,c1,c2)
    d=raw[:,1:]-raw[:,:-1]; dm,ds=p10.stats(d); pc1,pc2=p10.fit_rvq(p10.flat2(p10.norm5(d,dm,ds)),16,4,103,104,iters=2,max_points=1000); plain=(dm,ds,pc1,pc2)
    p10.encode_state_seq(raw[:,:3],state); p10.encode_plain_seq(raw[:,:3],state,plain); p10.encode_comp_seq(raw[:,:3],state,dm,ds)
    np.testing.assert_array_equal(raw,before)
