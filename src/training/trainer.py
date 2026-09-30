#!/usr/bin/env python
"""Single-GPU joint framework: --mode check audits graphs; train is explicit.

Data currently bound to the verified q v2 pool (598 train / 99 diagnostic val).
This entry does not silently expand the data pool or implement DDP / official RL.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch

from src.utils.paths import ROOT
from src.utils.hashing import sha256_file
from src.model.shared import forward as joint_forward
from src.model.future.targets import NativeFutureTargets
from src.data.q_targets import DEFAULT_Q_ARTIFACT,target_binding
from src.training.runtime import JointRuntime
from src.model.joint import ThreeBranchSystem,JointLossWeights
from src.training.checkpoint import save_joint,load_joint
from src.training.evaluation import evaluate, predict

def group_gradients(groups):
    return {name:{'tensors':len(params),'with_grad':sum(p.grad is not None for p in params),
        'nonzero':sum(p.grad is not None and bool(p.grad.abs().sum()>0) for p in params),
        'finite':all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in params)}
        for name,params in groups.items()}

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--mode',choices=('check','train','infer'),default='check')
    p.add_argument('--steps',type=int,default=150)
    p.add_argument('--checkpoint-every',type=int,default=50)
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--val-scenes',type=int,default=10,help='fixed eligible val subset; 0 = all')
    p.add_argument('--max-new-tokens',type=int,default=256)
    p.add_argument('--generated-val',action='store_true',help='also evaluate self-generated Reason at checkpoints')
    p.add_argument('--lr',type=float,default=1e-4)
    p.add_argument('--planner-lr',type=float,default=1e-5)
    p.add_argument('--lambda-text',type=float,default=0.,help='text CE weight; disabled by default')
    p.add_argument('--lambda-kl',type=float,default=1.)
    p.add_argument('--lambda-future',type=float,default=1.)
    p.add_argument('--lambda-planning',type=float,default=1.)
    p.add_argument('--q-artifact',type=Path,default=DEFAULT_Q_ARTIFACT)
    p.add_argument('--planner',type=Path,default=ROOT/'model/planner-rl')
    p.add_argument('--resume',type=Path)
    p.add_argument('--max-scenes',type=int,default=0)
    p.add_argument('--output-dir',type=Path,required=True)
    args=p.parse_args()
    if args.output_dir.exists():p.error('output directory already exists')
    if args.steps<1 or args.checkpoint_every<1 or args.max_scenes<0:p.error('invalid step/pool count')
    if min(args.lambda_text,args.lambda_kl,args.lambda_future,args.lambda_planning)<0:p.error('negative loss weight')
    if not (args.planner/'model.safetensors').is_file():p.error('missing planner weights')
    if args.val_scenes < 0 or args.max_new_tokens < 1:p.error('invalid validation/generation count')
    if not np.isfinite([args.lr,args.planner_lr,args.lambda_text,args.lambda_kl,args.lambda_future,args.lambda_planning]).all() or min(args.lr,args.planner_lr)<=0:p.error('invalid LR/loss weight')
    if args.mode=='infer' and args.resume is None:p.error('infer requires --resume CHECKPOINT')
    if args.mode=='check' and args.resume is not None:p.error('check cannot resume')
    if not torch.cuda.is_available():p.error('CUDA GPU required')
    run=JointRuntime(args)              # installs LoRA / differentiable shared forward
    run.model.load_planner(args.planner)  # critical: do not train a random uninitialised planner
    torch.manual_seed(args.seed)
    system=ThreeBranchSystem(run.model,run.forward,run.head).to('cuda').train()
    groups=system.parameter_groups()
    flat=[v for group in groups.values() for v in group]
    if len({id(v) for v in flat})!=len(flat):raise ValueError('duplicated optimizer parameters')
    if {id(v) for v in system.parameters() if v.requires_grad}!={id(v) for v in flat}:
        raise ValueError('trainable parameters do not match optimizer groups')
    optimizer=None if args.mode=='infer' else torch.optim.AdamW([
        {'params':params,'lr':args.planner_lr if name=='planner' else args.lr,'name':name}
        for name,params in groups.items()])
    weights=JointLossWeights(args.lambda_text,args.lambda_kl,args.lambda_future,args.lambda_planning)
    pool=run.train_rows[:args.max_scenes] if args.max_scenes else run.train_rows
    files=tuple(str(p.relative_to(ROOT)) for p in sorted((ROOT/'src').rglob('*.py')))
    protocol={'kind':'three_branch_shared_kv_v3','seed':args.seed,'loss_weights':asdict(weights),
        'lr':args.lr,'planner_lr':args.planner_lr,'pool_tokens':[run.train_tokens[i] for i in pool],
        'q_artifact':target_binding(run.q['train']),
        'q_val_artifact':target_binding(run.q['val']),
        'sequence_contract':joint_forward.CONTRACT_VERSION,
        'lora_parameter_dtype':'float32',
        'validation_tokens':[run.datasets['val'].tokens[i] for i in
            np.flatnonzero(run.q['val']['arrays']['record_status']==0).tolist()[:args.val_scenes or None]],
        'generated_val':args.generated_val,'max_new_tokens':args.max_new_tokens,
        'planner_sha256':sha256_file(args.planner/'model.safetensors'),
        'base_weights_sha256':sha256_file(ROOT/'model/model.safetensors'),
        'model_config_sha256':sha256_file(ROOT/'model/config.json'),
        'native_manifest_sha256':sha256_file(ROOT/'data/native/manifest.json'),
        'reason_text_manifest_sha256':sha256_file(ROOT/'data/reason_text/manifest.json'),
        'code_sha256':{n:sha256_file(ROOT/n) for n in files},
        'future_config':asdict(system.future.config),
        'branch_link':'shared differentiable S+T KV only; no Future-to-Planning connector',
        'planning_loss':'masked normalized clean-endpoint MSE, circular heading residual',
        'noise':'normal*config.noise_init_std','flow_time':'uniform[0,1)',
        'training_scope':'single GPU B=1; current verified q pool; supervised, no RL'}
    # Dataclass tuples must use the same JSON-safe representation in saved checkpoints.
    protocol=json.loads(json.dumps(protocol))
    if args.mode=='infer':
        saved=torch.load(args.resume,map_location='cpu',weights_only=True)
        saved_protocol=saved.get('protocol',{})
        # Inference can change subset/seed/LR flags, but never model/data/code binding.
        identity=('kind','sequence_contract','lora_parameter_dtype','q_artifact','q_val_artifact',
            'planner_sha256','base_weights_sha256','model_config_sha256',
            'native_manifest_sha256','reason_text_manifest_sha256','code_sha256','future_config')
        if any(saved_protocol.get(k)!=protocol.get(k) for k in identity):
            raise ValueError('inference checkpoint model/data/code/contract mismatch')
        del saved
        start=load_joint(args.resume,system,protocol=saved_protocol)
    else:
        start=load_joint(args.resume,system,optimizer,protocol=protocol) if args.resume else 0
    if args.mode=='train' and start>=args.steps:p.error('--steps must exceed resumed step')
    args.output_dir.mkdir(parents=True)
    (args.output_dir/'protocol.json').write_text(json.dumps(protocol,indent=2))
    if args.mode=='infer':
        result=predict(run,system,limit=args.val_scenes,seed=args.seed,
            max_new_tokens=args.max_new_tokens,output_dir=args.output_dir/'predictions')
        result['checkpoint_step']=start
        (args.output_dir/'report.json').write_text(json.dumps(result,indent=2))
        print(json.dumps({'scene_count':result['scene_count'],'checkpoint_step':start}),flush=True)
        return
    teacher=NativeFutureTargets(run.model)
    report={'kind':'three_branch_check' if args.mode=='check' else 'three_branch_train',
            'parameter_groups':{n:sum(v.numel() for v in ps) for n,ps in groups.items()},
            'history':[],'validation':[],'protocol':protocol,'checks':{}}
    length=1 if args.mode=='check' else args.steps
    torch.cuda.reset_peak_memory_stats()
    for step in range(start,length):
        epoch,offset=divmod(step,len(pool))
        row=int(np.random.default_rng(args.seed+epoch).permutation(pool)[offset])
        item=run.train_item(row)
        prompt=item['inputs']['input_ids'][0].tolist()
        sequence=joint_forward.build_sequence(prompt,run.t_pred['train'][item['token']],run.table,
                                               expected_token=item['token'])
        language=joint_forward.build_language_targets(sequence)
        inputs={k:v.to('cuda') if torch.is_tensor(v) else v for k,v in item['inputs'].items()}
        # Only the fixed visual tower is called for the GT; no VLM text forward here.
        pairing={**item['future_frames'],'token':item['token']}
        target=teacher.encode_scene(pairing,image_root=run.datasets['train'].image_root)
        gt=torch.as_tensor(item['trajectory']['future_traj_10hz'],device='cuda').unsqueeze(0)
        mask=torch.as_tensor(item['trajectory']['future_valid_mask_10hz'],device='cuda').unsqueeze(0)
        candidate_gt=run.q['train']['arrays']['trajectories'][row,run.targets['train']['gt_index'][row]]
        if not np.allclose(candidate_gt,item['trajectory']['future_traj_10hz'],atol=1e-5,rtol=0):
            raise ValueError('q GT and planning GT differ')
        supervision={'q':run.targets['train']['q'][row:row+1],
            'candidate_valid':run.targets['train']['valid'][row:row+1],
            'future_features':target['features'].unsqueeze(0).to('cuda'),
            'trajectory':gt,'trajectory_valid':mask}
        generator=torch.Generator(device='cuda').manual_seed(args.seed*1000000+step)
        noise=torch.randn(gt.shape,generator=generator,device='cuda')*run.model.config.noise_init_std
        time=torch.rand(1,generator=generator,device='cuda')
        with torch.autocast('cuda',dtype=torch.bfloat16):
            condition=system.condition(sequence=sequence,inputs=inputs,language=language,
                candidate_features=run.features['train'][row:row+1],candidate_valid=supervision['candidate_valid'])
            if condition['grid']!=target['grid']:raise ValueError('observed and target grid differ')
            losses=system.losses(condition,inputs=inputs,language=language,supervision=supervision,
                                 noise=noise,flow_time=time,weights=weights)
        if not torch.isfinite(losses['total']):raise ValueError('nonfinite total loss')
        if args.mode=='check':
            expected={'text':{'vlm_lora'},'reason_kl':{'vlm_lora','reason_head'},
                'future':{'vlm_lora','future'},'planning':{'vlm_lora','planner'}}
            report['per_loss_gradients']={}
            for key,value in condition['scene_kv']:
                key.retain_grad();value.retain_grad()
            report['checks']['same_planning_kv_objects']=condition['planning_kv'] is condition['scene_kv']
            report['checks']['same_planning_anchor_object']=condition['planning_anchor'] is condition['anchor']
            report['checks']['no_connector_parameters']=not hasattr(system,'memory') and set(groups)=={'vlm_lora','reason_head','future','planner'}
            for name,loss in losses['losses'].items():
                system.zero_grad(set_to_none=True)
                for key,value in condition['scene_kv']:key.grad=None;value.grad=None
                loss.backward(retain_graph=True)
                if name in ('future','planning'):
                    report['checks'][f'{name}_shared_kv_gradient']=any(
                        t.grad is not None and bool(t.grad.abs().sum()>0)
                        for pair in condition['scene_kv'] for t in pair)
                audit=group_gradients(groups);report['per_loss_gradients'][name]=audit
                active={n for n,v in audit.items() if v['nonzero']>0}
                report['checks'][f'{name}_gradient_routing']=active==expected[name] and all(v['finite'] for v in audit.values())
                if name=='planning':
                    report['checks']['planning_does_not_train_future_head']=all(p.grad is None for p in system.future.parameters())
                report['checks'][f'{name}_frozen_zero_grad']=all(v.grad is None for v in system.parameters() if not v.requires_grad)
            report['checks']['one_vlm_forward']=condition['vlm_calls']==1
            report['checks']['future_native_shape']=list(condition['future_prediction'].shape)==[1,*target['features'].shape]
            report['checks']['gt_teacher_stop_gradient']=not target['features'].requires_grad
        system.zero_grad(set_to_none=True);losses['total'].backward()
        grads=group_gradients(groups)
        if not all(v['finite'] for v in grads.values()):raise ValueError('nonfinite gradient')
        torch.nn.utils.clip_grad_norm_(flat,1.)
        if args.mode=='check':
            lora_before=[p.detach().clone() for p in groups['vlm_lora']]
        optimizer.step()
        if args.mode=='check':
            report['checks']['lora_params_fp32']=all(p.dtype==torch.float32 for p in groups['vlm_lora'])
            report['checks']['lora_adam_states_fp32']=all(
                optimizer.state[p]['exp_avg'].dtype==optimizer.state[p]['exp_avg_sq'].dtype==torch.float32
                for p in groups['vlm_lora'])
            report['checks']['lora_actual_update']=any(not torch.equal(a,p) for a,p in zip(lora_before,groups['vlm_lora']))
            del lora_before
        report['history'].append({'step':step+1,'token':item['token'],'vlm_calls':condition['vlm_calls'],
            'losses':{n:float(v.detach()) for n,v in losses['losses'].items()},
            'total':float(losses['total'].detach()),'future_shape':list(condition['future_prediction'].shape),
            'planner_endpoint_shape':list(losses['planning']['trajectory'].shape),
            'gradients':grads,'flow_time':float(time.item())})
        print(json.dumps(report['history'][-1]),flush=True)
        del condition,losses,supervision,target
        optimizer.zero_grad(set_to_none=True)
        if args.mode=='check' or (step+1)%args.checkpoint_every==0 or step+1==args.steps:
            limit=min(args.val_scenes or 2,2) if args.mode=='check' else args.val_scenes
            for source in ('pseudo_label','generated') if args.generated_val else ('pseudo_label',):
                validation=evaluate(run,system,weights=weights,limit=limit,seed=args.seed,
                    reason_source=source,max_new_tokens=args.max_new_tokens)
                validation['step']=step+1
                report['validation'].append(validation)
                folder=args.output_dir/'validation';folder.mkdir(exist_ok=True)
                (folder/f'step_{step+1:06d}_{source}.json').write_text(json.dumps(validation,indent=2))
                print(json.dumps({'step':step+1,'validation':source,
                    'mean_losses':validation['mean_losses'],'mean_metrics':validation['mean_metrics']}),flush=True)
            if args.mode=='train':
                save_joint(args.output_dir/f'checkpoints/step_{step+1:06d}.pt',system,optimizer,step=step+1,protocol=protocol)
    report['peak_gpu_gib']=torch.cuda.max_memory_allocated()/1024**3
    report['completed_steps']=start+len(report['history'])
    report['ok']=all(report['checks'].values()) if args.mode=='check' else report['completed_steps']==args.steps
    report['quality_claim']='execution and finite held-out metrics only; no planning improvement gate' 
    (args.output_dir/'report.json').write_text(json.dumps(report,indent=2))
    raise SystemExit(0 if report['ok'] else 1)

if __name__=='__main__':main()
