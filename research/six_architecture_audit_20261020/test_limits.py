"""Boundary checks for the user's two hard ceilings."""
import time
import pytest
import torch
from torch import nn
import run_pilots
from conv7_language import ScheduledLM,count

def test_next_batch_cannot_cross_two_billion():
    run=run_pilots.Attempt.__new__(run_pilots.Attempt)
    run.tokens=2_000_000_000-16;run.started=time.perf_counter()
    run.expose(16)
    assert run.tokens==2_000_000_000
    with pytest.raises(ValueError,match='TOKEN_CAP'):run.expose(1)
    assert run.tokens==2_000_000_000

def test_model_above_one_hundred_million_rejected_before_attempt(tmp_path):
    with torch.device('meta'):
        model=nn.Embedding(100_000_001,1)
    with pytest.raises(ValueError,match='PARAMETER_CAP'):
        run_pilots.Attempt(tmp_path,'oversize',model,{})
    assert not list(tmp_path.iterdir())

def test_scaled_conv7_counts_are_actually_under_cap():
    with torch.device('meta'):
        models=[ScheduledLM(504,n) for n in ('transformer','attention8','conv7')]
    counts=[count(m) for m in models]
    assert counts==[99_051_624,99_059_688,99_043_576]
    assert max(counts)<100_000_000
    assert abs(counts[2]-counts[0])/counts[0]<.001
