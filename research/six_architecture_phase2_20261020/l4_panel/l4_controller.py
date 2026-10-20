"""Fresh L4 paired panel with reserved segments and acknowledged full backup."""
import hashlib,json,math,time,traceback
from pathlib import Path
import torch
import train_panel as panel

ROOT=Path('/content/tam-l4-panel-20261020');OUT=ROOT/'results';DATA=ROOT/'data'
panel.ROOT=ROOT;panel.OUT=OUT;panel.DATA=DATA;panel.SEED=202610321
SCIENTIFIC=panel.STAGES;UNIT=panel.UNIT
MAX_SEGMENT=25_000_000//UNIT

def sha(p):return panel.sha(p)
def atomic(p,v):panel.atomic(p,v)
def wait_ack(name,digest):
    start=time.monotonic()
    while time.monotonic()-start<3600:
        p=ROOT/name
        if p.exists() and json.loads(p.read_text()).get('digest')==digest:return
        time.sleep(5)
    raise TimeoutError('Local durable acknowledgement missing for60minutes')

def main():
    torch.set_num_threads(4);OUT.mkdir(exist_ok=True)
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    exclusive=ROOT/'L4_CLAIM.json'
    panel.exclusive(exclusive,{'seed':panel.SEED,'protocol_sha256':sha(ROOT/'RECOVERY_PROTOCOL.md'),'created_utc':time.time()})
    while not (DATA/'DATA.json').exists():panel.status(phase='waiting_for_corpus',complete=False);time.sleep(10)
    data=json.loads((DATA/'DATA.json').read_text())
    for split in ('train','validation','test'):assert sha(DATA/(split+'.bin'))==data['splits'][split]['token_sha256']
    evidence={'trial':'L4_NEW_PAIRED_SEED','seed':panel.SEED,'data':data,'results':{},'gates':[],'terminal':{},'complete':False,
        'latest':{},'charges':{a:0 for a in ('transformer','attention8','conv7','cpw_local')},
        'controller_source_sha256':sha(Path(__file__)),'recovery_protocol_sha256':sha(ROOT/'RECOVERY_PROTOCOL.md')}
    active=['transformer','attention8','conv7','cpw_local'];start_stage=0
    if (ROOT/'RESUME.json').exists():
        resume=json.loads((ROOT/'RESUME.json').read_text());evidence=resume['evidence']
        assert evidence['seed']==panel.SEED and evidence['controller_source_sha256']==sha(Path(__file__))
        assert evidence['data']['splits']==data['splits']
        active=resume['active_arms'];start_stage=resume['stage_index']
        assert all(type(n) is int and 0<=n<=2_000_000_000 for n in evidence['charges'].values())
    previous=dict(evidence['latest'])
    latest={arm:evidence['results'][f"{arm}-s{m['step']}"] for arm,m in previous.items()}
    atomic(ROOT/'EVIDENCE.json',evidence)
    for stage_index in range(start_stage,len(SCIENTIFIC)):
        desired=SCIENTIFIC[stage_index]
        target=min(desired,*[(2_000_000_000-evidence['charges'][arm])//UNIT for arm in active])
        cap_terminal=target<desired
        evidence['active_arms']=active;evidence['stage_index']=stage_index
        while any(previous.get(a,{}).get('step',0)<target for a in active):
            for arm in active:
                start=previous.get(arm,{}).get('step',0)
                if start>=target:continue
                # Redistribute the interval to avoid a tiny trailing segment.
                remaining=target-start;pieces=math.ceil(remaining/MAX_SEGMENT);end=start+math.ceil(remaining/pieces)
                ident=f'{arm}-s{end}'
                reservation={'id':ident,'variant':arm,'seed':panel.SEED,'start':start,'end':end,
                    'reserved_exposures':(end-start)*UNIT,'charged_lost_exposures':evidence['charges'][arm],
                    'cumulative_upper_bound':end*UNIT+evidence['charges'][arm],'controller_source_sha256':sha(Path(__file__))}
                assert reservation['cumulative_upper_bound']<=2_000_000_000
                digest=hashlib.sha256(json.dumps(reservation,sort_keys=True).encode()).hexdigest();reservation['digest']=digest
                atomic(ROOT/'ACTIVE_SEGMENT.json',reservation);panel.status(phase='awaiting_reservation_backup',current=arm,complete=False)
                wait_ack('RESERVATION_ACK.json',digest)
                result=panel.segment(arm,end,previous.get(arm),evidence)
                result['trial']='L4_NEW_PAIRED_SEED';result['scientific_endpoint']=end==target
                result['charged_lost_exposures']=evidence['charges'][arm]
                result['controller_source_sha256']=sha(Path(__file__))
                evidence['results'][ident]=result;evidence['latest'][arm]=result['checkpoint']
                atomic(ROOT/'EVIDENCE.json',evidence)
                panel.status(phase='awaiting_checkpoint_backup',current=arm,step=end,complete=False)
                wait_ack('CHECKPOINT_ACK.json',result['checkpoint']['checkpoint_sha256'])
                previous[arm]=result['checkpoint'];latest[arm]=result
        control=latest['transformer'];keep=[]
        for arm in ('conv7','cpw_local'):
            if arm not in active:continue
            r=latest[arm];delta=r['validation']['nll']-control['validation']['nll'];speed=r['train_tokens_per_second']/control['train_tokens_per_second']
            benefit=latest['attention8']['validation']['nll']-r['validation']['nll'] if arm=='conv7' else None
            passed=((delta<=.05 or (delta<=.10 and speed>=1.03)) if stage_index==0 else (delta<=-.03 or (delta<=.015 and speed>=1.03)))
            if benefit is not None:passed=passed and benefit>=.005
            evidence['gates'].append({'stage_tokens':target*UNIT,'variant':arm,'delta_nll_to_transformer':delta,'speed_ratio':speed,'parent_benefit':benefit,'passed':passed})
            if passed and stage_index<len(SCIENTIFIC)-1 and not cap_terminal:keep.append(arm)
            elif arm not in evidence['terminal']:panel.final_test(arm,r,evidence)
        if 'attention8' in active and 'conv7' not in keep and 'attention8' not in evidence['terminal']:panel.final_test('attention8',latest['attention8'],evidence)
        atomic(ROOT/'EVIDENCE.json',evidence)
        if not keep:
            if 'transformer' not in evidence['terminal']:panel.final_test('transformer',control,evidence)
            break
        active=['transformer']+(['attention8'] if 'conv7' in keep else [])+keep
        evidence['active_arms']=active;evidence['stage_index']=stage_index+1;atomic(ROOT/'EVIDENCE.json',evidence)
    evidence['complete']=True;atomic(ROOT/'EVIDENCE.json',evidence);panel.status(phase='finished',complete=True)

if __name__=='__main__':
    try:main()
    except BaseException as e:panel.status(phase='error',complete=True,error=repr(e),traceback=traceback.format_exc());raise
