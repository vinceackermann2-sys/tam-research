"""Integrated functional interpretation of the user's original SI architecture.

This is NOT the scientifically tested complete system or an SI breakthrough.
The screenshots specify functions, not a unique build. Choices here are
explicit ENGINEERING INTERPRETATIONS: small numerical token encoders, original
attention/recurrent/sparse-MoE blocks, bounded latent halting, genuine two-step
action search, public-only verified persistent memory, online rank-two fast
weights, replay, modular skill adapters and a decision-gating verifier.

All component paths are actually called. Training, scaling, world-model
accuracy, true native multimodal competence and cross-domain generalization
remain UNPROVEN. No hidden-state oracle, scientific seeds, or GPU required.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from typing import Any, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from tam_research.si_original_hybrid_v0 import SIConfig, SIHybridBlock

KINDS = ("text", "image", "audio", "video", "motor")
PROVENANCE = "executed_public_feedback"


@dataclass(frozen=True)
class PublicExperience:
    world: str
    skill: str | None
    before: torch.Tensor
    action: int
    reward: float
    after: torch.Tensor
    provenance: str


class VersionedVerifiedMemory:
    """Detached causal transition deltas; JSON-exportable, world-scoped."""

    def __init__(self, width: int, capacity: int = 128):
        if width < 1 or capacity < 1:
            raise ValueError("positive memory dimensions")
        self.width, self.capacity = width, capacity
        self.rows: deque[tuple[str, torch.Tensor, torch.Tensor]] = deque(maxlen=capacity)

    def write(self, world: str, key: torch.Tensor, value: torch.Tensor,
              *, provenance: str) -> None:
        if not world or provenance != PROVENANCE:
            raise ValueError("verified executed feedback required")
        if (key.shape != (self.width,) or value.shape != (self.width,)
                or not torch.isfinite(key).all() or not torch.isfinite(value).all()):
            raise ValueError("invalid fact representation")
        self.rows.append((world, key.detach().cpu().clone(), value.detach().cpu().clone()))

    def read(self, world: str, query: torch.Tensor) -> torch.Tensor:
        eligible = [(k, v) for w, k, v in self.rows if w == world]
        if not eligible:
            return torch.zeros_like(query)
        if query.ndim != 1 or query.numel() != self.width:
            raise ValueError("invalid memory query")
        keys = torch.stack([key.to(query.device) for key, _ in eligible])
        match = F.cosine_similarity(keys, query.detach()[None], dim=-1)
        return eligible[int(match.argmax())][1].to(query.device).to(query.dtype)

    def dump(self) -> list[dict[str, Any]]:
        return [{"world": w, "key": k.tolist(), "value": v.tolist(),
                 "provenance": PROVENANCE} for w, k, v in self.rows]

    def load(self, records: list[dict[str, Any]]) -> None:
        if type(records) is not list or len(records) > self.capacity:
            raise ValueError("invalid bounded verified memory export")
        checked = VersionedVerifiedMemory(self.width, self.capacity)
        for record in records:
            if (type(record) is not dict or
                    set(record) != {"world", "key", "value", "provenance"}):
                raise ValueError("invalid memory record schema")
            checked.write(
                record["world"], torch.tensor(record["key"], dtype=torch.float32),
                torch.tensor(record["value"], dtype=torch.float32),
                provenance=record["provenance"])
        self.rows = checked.rows


class SIOriginalIntegratedV1(nn.Module):
    """Small integrated vertical slice of ALL user-required mechanisms.

    It reuses the original source's actual alternating causal-attention,
    recurrent-state, sparsely-dispatched MoE implementation. Other mechanisms
    are wired functional interpretations, not validated scientific advances.
    """

    def __init__(self, *, width: int = 48, layers: int = 4, experts: int = 4,
                 max_steps: int = 3, memory_capacity: int = 128,
                 replay_capacity: int = 128):
        super().__init__()
        if (width < 16 or width % 4 or layers < 2 or experts < 2
                or not 2 <= max_steps <= 8
                or memory_capacity < 1 or replay_capacity < 1):
            raise ValueError("invalid bounded integrated architecture")
        self.width = width
        self.max_steps = max_steps
        self.cfg = SIConfig(obs_dim=3, action_dim=4, d_model=width,
                            layers=layers, heads=4, window=12,
                            experts=experts, max_seq_len=64)
        # REAL modality-specific learned feature extractors, not only tags.
        self.text_encoder = nn.Embedding(256, width)
        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, width//2, 3, padding=1), nn.GELU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(width//2, width))
        self.audio_encoder = nn.Sequential(
            nn.Conv1d(1, width//2, 3, padding=1), nn.GELU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(width//2, width))
        self.video_encoder = nn.Sequential(
            nn.Conv3d(3, width//2, (2,3,3), padding=(0,1,1)), nn.GELU(),
            nn.AdaptiveAvgPool3d(1), nn.Flatten(), nn.Linear(width//2, width))
        self.motor_encoder = nn.Linear(4, width)
        self.modality = nn.Embedding(len(KINDS), width)
        self.position = nn.Embedding(64, width)
        self.blocks = nn.ModuleList([
            SIHybridBlock(self.cfg, i) for i in range(layers)])
        self.norm = nn.LayerNorm(width)

        # ACT-style physically variable number of latent iterative steps.
        self.latent_step = nn.GRUCell(width, width)
        self.halt = nn.Linear(width, 1)
        self.halt_bias = nn.Parameter(torch.tensor(0.0))
        self.action_head = nn.Linear(width, 4)
        self.transition = nn.Sequential(
            nn.Linear(width+4, width), nn.GELU(), nn.Linear(width,width))
        self.reward_head = nn.Linear(width, 1)
        self.value_head = nn.Linear(width, 1)
        self.verifier = nn.Sequential(nn.Linear(width, width//2),
                                      nn.GELU(), nn.Linear(width//2, 1))
        self.memory_gate = nn.Parameter(torch.tensor(0.0))
        self.memory = VersionedVerifiedMemory(width, memory_capacity)
        self.replay: deque[PublicExperience] = deque(maxlen=replay_capacity)
        self.skills = nn.ModuleDict()

        # Inference-time low-rank fast weights are BUFFERS updated only from
        # factual rewards, never from hidden teacher/counterfactual labels.
        init = torch.linspace(-.5, .5, width*2).reshape(width,2)
        self.register_buffer("fast_u", init.clone())
        self.register_buffer("fast_v", torch.zeros(2))
        self.fast_direction = nn.Parameter(torch.ones(width) / math.sqrt(width))
        self.fast_updates = 0
        self.latest_replay_loss: float | None = None

    def add_skill(self, name: str) -> None:
        if (not isinstance(name,str) or not name or
                not name.replace("_","").isalnum() or name in self.skills):
            raise ValueError("new named modular skill required")
        adapter = nn.Sequential(nn.Linear(self.width, 8), nn.Tanh(),
                                nn.Linear(8, self.width, bias=False))
        nn.init.zeros_(adapter[-1].weight)
        self.skills[name] = adapter

    def _encode_one(self, kind: str, value: torch.Tensor) -> torch.Tensor:
        if kind not in KINDS or not isinstance(value,torch.Tensor):
            raise ValueError("unsupported public modality")
        if kind == "text":
            if value.ndim != 1 or value.dtype not in (torch.int32, torch.int64):
                raise ValueError("text tokens must be B integer IDs")
            if bool(((value<0)|(value>=256)).any()):
                raise ValueError("token out of vocabulary")
            return self.text_encoder(value.long())
        if not value.is_floating_point() or not bool(torch.isfinite(value).all()):
            raise ValueError("finite real-valued public sensor required")
        if kind == "image":
            if value.ndim != 4 or value.shape[1:] != (3,8,8):
                raise ValueError("expected B,3,8,8 public image")
            return self.image_encoder(value)
        if kind == "audio":
            if value.ndim != 2 or value.shape[1] != 32:
                raise ValueError("expected B,32 public audio")
            return self.audio_encoder(value[:,None])
        if kind == "video":
            if value.ndim != 5 or value.shape[1:] != (2,3,8,8):
                raise ValueError("expected B,2,3,8,8 public video")
            return self.video_encoder(value.permute(0,2,1,3,4))
        if value.ndim != 2 or value.shape[1] != 4:
            raise ValueError("expected B,4 actual motor vector")
        return self.motor_encoder(value)

    def encode(self, steps: Sequence[tuple[str, torch.Tensor]]) -> torch.Tensor:
        if not 1 <= len(steps) <= 64:
            raise ValueError("context length must be within 1..64")
        items = []
        batch: int | None = None
        for i,(kind, value) in enumerate(steps):
            hidden=self._encode_one(kind,value)
            if batch is None:
                batch=hidden.size(0)
            if hidden.size(0) != batch or batch == 0:
                raise ValueError("inconsistent multimodal batch sizes")
            tag=torch.tensor(KINDS.index(kind),device=hidden.device)
            pos=torch.tensor(i,device=hidden.device)
            items.append(hidden+self.modality(tag)+self.position(pos))
        return torch.stack(items,dim=1)

    def _backbone(self, steps: Sequence[tuple[str, torch.Tensor]]
                  ) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        seq=self.encode(steps)
        balances=[]
        routes=[]
        for block in self.blocks:
            seq, bal, route = block(seq)
            balances.append(bal)
            if route is not None:
                routes.append(route)
        return self.norm(seq[:,-1]),torch.stack(balances).sum(),routes

    def _fast_correction(self, state: torch.Tensor) -> torch.Tensor:
        a = F.normalize(state, dim=-1)
        residual = (a @ self.fast_u) @ self.fast_v
        return state + residual[:,None]*self.fast_direction[None]

    def _reason(self,state:torch.Tensor, *, steps_limit:int
                ) -> tuple[torch.Tensor,torch.Tensor]:
        current=state
        active=torch.ones(state.size(0),dtype=torch.bool,device=state.device)
        counts=torch.zeros(state.size(0),dtype=torch.int64,device=state.device)
        for _ in range(steps_limit):
            if not bool(active.any()):
                break
            index=torch.nonzero(active,as_tuple=False).flatten()
            updated=self.latent_step(state[index],current[index])
            current=current.index_copy(0,index,updated)
            counts[index]+=1
            probability=torch.sigmoid(self.halt(current[index]).flatten()+self.halt_bias)
            active[index]=probability<=.5
        return current,counts

    def _world_step(self,state:torch.Tensor,action:int
                    ) -> tuple[torch.Tensor,torch.Tensor]:
        if type(action) is not int or action not in range(4):
            raise ValueError("legal motor action 0..3 required")
        one_hot=F.one_hot(torch.full((state.size(0),),action,
                           device=state.device),4).to(state.dtype)
        nxt=state+self.transition(torch.cat([state,one_hot],-1))
        reward=self.reward_head(nxt).flatten()
        return nxt,reward

    @torch.no_grad()
    def _search(self,state:torch.Tensor) -> tuple[torch.Tensor,int]:
        """ACTUAL 2-step depth-2 exhaustive tree search, 4 + 16 edges."""
        scores=[]
        expansions=0
        for first in range(4):
            after,rew1=self._world_step(state,first)
            candidate=[]
            expansions+=1
            for second in range(4):
                future,rew2=self._world_step(after,second)
                value=rew1+0.9*rew2+0.81*self.value_head(future).flatten()
                candidate.append(value)
                expansions+=1
            scores.append(torch.stack(candidate,dim=-1).max(dim=-1).values)
        return torch.stack(scores,dim=-1).argmax(dim=-1),expansions

    def forward(self,steps:Sequence[tuple[str,torch.Tensor]], *,
                world:str="default",skill:str|None=None,
                enable_memory:bool=True,enable_fast:bool=True,
                enable_search:bool=True,uncertainty_threshold:float=.5,
                reason_budget:int|None=None) -> dict[str,Any]:
        if (not world or (skill is not None and skill not in self.skills)
                or not 0 <= uncertainty_threshold <= 1):
            raise ValueError("unknown world/skill or verifier threshold")
        limit=self.max_steps if reason_budget is None else reason_budget
        if type(limit) is not int or not 1 <= limit <= self.max_steps:
            raise ValueError("bounded latent compute required")
        state,balance,routes=self._backbone(steps)
        # Long-term retrieval enters BEFORE iterative reasoning/planning.
        memory_value=[]
        for i in range(state.size(0)):
            memory_value.append(self.memory.read(world,state[i])
                                if enable_memory else torch.zeros_like(state[i]))
        fetched=torch.stack(memory_value)
        state=state+torch.sigmoid(self.memory_gate)*fetched
        if enable_fast:
            state=self._fast_correction(state)
        if skill is not None:
            state=state+self.skills[skill](state)
        reasoned,iterations=self._reason(state,steps_limit=limit)
        logits=self.action_head(reasoned)
        uncertainty=torch.sigmoid(self.verifier(reasoned)).flatten()
        direct=logits.argmax(dim=-1)
        action=direct.clone()
        search_flags=uncertainty>uncertainty_threshold
        expansions=0
        if enable_search and bool(search_flags.any()):
            idx=search_flags.nonzero(as_tuple=False).flatten()
            chosen,count=self._search(reasoned[idx])
            action=action.index_copy(0,idx,chosen)
            expansions=count*len(idx)
        return {
            "action":action, "direct_action":direct, "logits":logits,
            "uncertainty":uncertainty, "searched":search_flags if enable_search
            else torch.zeros_like(search_flags),
            "search_expansions":expansions,
            "latent_steps":iterations, "state":reasoned,
            "retrieval":fetched, "expert_routes":routes, "balance":balance,
        }

    @torch.no_grad()
    def ingest_executed(self, before:Sequence[tuple[str,torch.Tensor]],
                        action:int,reward:float,
                        after:Sequence[tuple[str,torch.Tensor]], *,
                        world:str,skill:str|None=None,
                        provenance:str=PROVENANCE) -> None:
        """Only caller-attested PUBLIC real before/action/reward/after.

        A private label or simulated counterfactual MUST NOT be inserted as
        an executed memory record. This Python API validates structure and
        provenance labels; it is not an authenticated sandbox receipt.
        """
        if (not world or provenance != PROVENANCE or
                type(action) is not int or action not in range(4) or
                type(reward) not in (int,float) or
                not math.isfinite(reward) or not -1 <= reward <= 1 or
                (skill is not None and skill not in self.skills)):
            raise ValueError("executed public experience required")
        if len(after)!=len(before)+1:
            raise ValueError("one actual next observation required")
        b,_,_=self._backbone(before)
        nxt,_,_=self._backbone(after)
        if b.shape != (1,self.width) or nxt.shape != b.shape:
            raise ValueError("single verified public trajectory required")
        prior=b[0].detach()
        later=nxt[0].detach()
        self.memory.write(world,prior,later-prior,provenance=provenance)
        self.replay.append(PublicExperience(
            world,skill,prior.cpu().clone(),action,float(reward),
            later.cpu().clone(),provenance))
        x=F.normalize(prior,dim=-1)
        z=x @ self.fast_u
        err=max(-1.,min(1.,reward-float(z @ self.fast_v)))
        oldv=self.fast_v.clone()
        self.fast_v.add_(.3*err*z).clamp_(-2,2)
        self.fast_u.add_(.3*err*x[:,None]*oldv[None]).clamp_(-2,2)
        self.fast_updates+=1

    def consolidate_verified_replay(self, optimizer:torch.optim.Optimizer,
                                    *, count:int=8) -> float:
        """Learn a predictive world/reward/verifier from *actual* replay."""
        if (not self.replay or type(count) is not int or not 1<=count<=32):
            raise ValueError("bounded factual replay required")
        batch=list(self.replay)[-count:]
        device=next(self.parameters()).device
        before=torch.stack([r.before for r in batch]).to(device)
        next_real=torch.stack([r.after for r in batch]).to(device)
        act=F.one_hot(torch.tensor([r.action for r in batch],device=device),4).float()
        rewards=torch.tensor([r.reward for r in batch],device=device).float()
        delta=self.transition(torch.cat([before,act],-1))
        predicted=before+delta
        predicted_reward=self.reward_head(predicted).flatten()
        mse=(predicted-next_real).square().mean(dim=-1)
        unc=torch.sigmoid(self.verifier(predicted)).flatten()
        calibration=(unc-mse.detach().clamp(0,1)).square().mean()
        loss=mse.mean()+F.mse_loss(predicted_reward,rewards)+.1*calibration
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        self.latest_replay_loss=float(loss.detach())
        return self.latest_replay_loss

    def export_persistent_state(self) -> dict[str,Any]:
        return {"schema":"si-original-integrated-v1/1",
                "width":self.width,"memory":self.memory.dump(),
                "fast_u":self.fast_u.detach().cpu().tolist(),
                "fast_v":self.fast_v.detach().cpu().tolist(),
                "fast_updates":self.fast_updates}

    @torch.no_grad()
    def import_persistent_state(self,record:dict[str,Any]) -> None:
        if (type(record) is not dict or
                set(record)!={"schema","width","memory","fast_u","fast_v","fast_updates"}
                or record["schema"]!="si-original-integrated-v1/1"
                or record["width"]!=self.width or
                type(record["fast_updates"]) is not int or record["fast_updates"]<0):
            raise ValueError("invalid serialized SI state")
        u=torch.tensor(record["fast_u"],dtype=self.fast_u.dtype,device=self.fast_u.device)
        v=torch.tensor(record["fast_v"],dtype=self.fast_v.dtype,device=self.fast_v.device)
        if (u.shape!=self.fast_u.shape or v.shape!=self.fast_v.shape
                or not bool(torch.isfinite(u).all())
                or not bool(torch.isfinite(v).all())):
            raise ValueError("invalid fast weights")
        checked=VersionedVerifiedMemory(self.width,self.memory.capacity)
        checked.load(record["memory"])
        self.fast_u.copy_(u)
        self.fast_v.copy_(v)
        self.memory=checked
        self.fast_updates=record["fast_updates"]
