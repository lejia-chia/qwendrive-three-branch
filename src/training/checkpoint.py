"""Save all trainable components without duplicating the frozen VLM backbone."""
from pathlib import Path
import os
import torch

def vlm_trainable(model):
    return {name:p for name,p in model.named_parameters()
            if p.requires_grad and not name.startswith('planning_expert.')}

def save_joint(path,system,optimizer,*,step,protocol):
    path=Path(path)
    if path.exists():raise FileExistsError(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    payload={'kind':'qwendrive_three_branch_shared_kv_v3','step':step,'protocol':protocol,
        'vlm_lora':{k:v.detach().cpu() for k,v in vlm_trainable(system.model).items()},
        'reason_head':system.reason_head.state_dict(),'future':system.future.state_dict(),
        'planner':system.model.planning_expert.state_dict(),
        'optimizer':optimizer.state_dict(),'torch_rng':torch.get_rng_state(),
        'cuda_rng':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}
    tmp=path.with_suffix(path.suffix+'.tmp')
    try:
        torch.save(payload,tmp);os.replace(tmp,path)
    finally:
        if tmp.exists():tmp.unlink()

def load_joint(path,system,optimizer=None,*,protocol):
    payload=torch.load(path,map_location='cpu',weights_only=True)
    if payload.get('kind')!='qwendrive_three_branch_shared_kv_v3' or payload.get('protocol')!=protocol:
        raise ValueError('checkpoint protocol/model/data/config mismatch')
    params=vlm_trainable(system.model)
    if set(payload['vlm_lora'])!=set(params):raise ValueError('LoRA parameter names differ')
    pieces={'reason_head':system.reason_head,'future':system.future,
            'planner':system.model.planning_expert}
    pairs=[(params[k],v) for k,v in payload['vlm_lora'].items()]
    for name,module in pieces.items():
        current=module.state_dict();saved=payload[name]
        if set(current)!=set(saved):raise ValueError(f'{name} parameter names differ')
        pairs.extend((current[k],v) for k,v in saved.items())
    if any(dst.shape!=src.shape or dst.dtype!=src.dtype or not torch.isfinite(src).all() for dst,src in pairs):
        raise ValueError('checkpoint tensor shape/dtype/nonfinite mismatch')
    with torch.no_grad():
        for name,p in params.items():p.copy_(payload['vlm_lora'][name].to(p.device,p.dtype))
    for name,module in pieces.items():module.load_state_dict(payload[name],strict=True)
    if optimizer is not None:
        optimizer.load_state_dict(payload['optimizer'])
        torch.set_rng_state(payload['torch_rng'])
        if payload['cuda_rng']:torch.cuda.set_rng_state_all(payload['cuda_rng'])
    return int(payload['step'])
