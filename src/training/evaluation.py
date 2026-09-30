"""Held-out evaluation and autoregressive three-branch prediction.

Training uses one teacher-forced main forward. Autoregressive inference first
runs generation from S, then one closed S+generated-T forward for all branches;
the hybrid recurrent cache is not reused as an ExportCache.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import torch

from src.model.shared.forward import build_sequence, content_sequence, build_language_targets
from src.model.shared.lora import modality_ids
from src.model.future.targets import NativeFutureTargets
from src.model.reasoning.scoring import kl_divergence


def device_inputs(item, device='cuda'):
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in item['inputs'].items()}


def prepared_sequence(run, item, split):
    sequence = build_sequence(item['inputs']['input_ids'][0].tolist(),
        run.t_pred[split][item['token']], run.table, expected_token=item['token'])
    return sequence, build_language_targets(sequence)


@torch.no_grad()
def generate_sequence(run, inputs, *, max_new_tokens=256):
    """Only current-observable processor tensors enter generation; no GT/text label."""
    if run.model.vlm.training:
        raise ValueError('generation requires eval mode')
    prompt = inputs['input_ids']
    with torch.autocast('cuda', dtype=torch.bfloat16):
        generated = run.model.vlm.generate(input_ids=prompt,
            pixel_values=inputs['pixel_values'], image_grid_thw=inputs['image_grid_thw'],
            mm_token_type_ids=modality_ids(run.model, prompt), max_new_tokens=max_new_tokens,
            min_new_tokens=run.model.config.min_reasoning_tokens, do_sample=False,
            num_beams=1, repetition_penalty=1.0, use_cache=True,
            return_dict_in_generate=True, eos_token_id=list(run.table.terminator_ids))
    ids = generated.sequences[0, prompt.shape[1]:].tolist()
    stop = next((i for i, token in enumerate(ids) if token in run.table.terminator_ids), None)
    if stop is None:
        raise ValueError(f'Reason did not stop within {max_new_tokens} tokens; truncated output rejected')
    content = ids[:stop]
    if any(token in run.table.forbidden_ids for token in content):
        raise ValueError('generated retired behavior token')
    sequence = content_sequence(prompt[0].tolist(), content, run.table)
    return sequence, build_language_targets(sequence), {
        'text': run.model.processor.tokenizer.decode(content, skip_special_tokens=True),
        'text_ids': content, 'stopped': True, 'sampled_terminator': ids[stop],
        'generation_tokens': stop + 1}


def trajectory_metrics(prediction, trajectory, valid):
    """XY-only metres; FDE at last valid point, with its actual horizon recorded."""
    prediction = prediction.detach().float().cpu().numpy()[0]
    gt = np.asarray(trajectory, dtype=np.float64)
    mask = np.asarray(valid)
    if prediction.shape != (50, 3) or gt.shape != prediction.shape or mask.shape != (50,):
        raise ValueError('metric trajectory shape mismatch')
    if not np.isin(mask, (0, 1)).all() or not np.isfinite(prediction).all() or not np.isfinite(gt[mask > 0]).all():
        raise ValueError('invalid metric trajectory/mask')
    result = {}
    for steps in (40, 50):
        live = np.flatnonzero(mask[:steps] > 0)
        if not len(live):
            raise ValueError('metric horizon has no valid GT')
        distance = np.linalg.norm(prediction[live, :2] - gt[live, :2], axis=-1)
        yaw = prediction[live, 2] - gt[live, 2]
        key = f'{steps // 10}s'
        result[f'ade_{key}_m'] = float(distance.mean())
        result[f'fde_{key}_m'] = float(distance[-1])
        result[f'fde_{key}_time_s'] = float((live[-1] + 1) * .1)
        result[f'yaw_mae_{key}_rad'] = float(np.abs(np.arctan2(np.sin(yaw), np.cos(yaw))).mean())
    return result


@torch.no_grad()
def evaluate(run, system, *, weights, limit=10, seed=0, reason_source='pseudo_label',
             max_new_tokens=256, prediction_dir=None):
    """Fixed eligible val rows; free-noise sampling measures trajectory quality.

    Endpoint loss at t=.5 is separately reported, never presented as sampling ADE.
    Future GT is accessed only AFTER conditioning. Per-scene metrics are averaged
    by scene; future MSE additionally records the element-weighted aggregate.
    """
    if reason_source not in ('pseudo_label', 'generated'):
        raise ValueError('unknown Reason source')
    rows = np.flatnonzero(run.q['val']['arrays']['record_status'] == 0).tolist()
    if limit > 0:
        rows = rows[:limit]
    if not rows:
        raise ValueError('no validation scenes')
    output = Path(prediction_dir) if prediction_dir is not None else None
    if output is not None:
        output.mkdir(parents=True, exist_ok=False)
    was_training = system.training
    teacher = NativeFutureTargets(run.model)
    records = []
    future_sse, future_count = 0., 0
    try:
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            system.eval()
            for row in rows:
                item = run.datasets['val'][row]
                inputs = device_inputs(item)
                if reason_source == 'generated':
                    sequence, language, reason = generate_sequence(run, inputs, max_new_tokens=max_new_tokens)
                else:
                    sequence, language = prepared_sequence(run, item, 'val')
                    reason = {'text_ids': run.t_pred['val'][item['token']]['text_ids'], 'source': 'frozen_pseudo_label'}
                candidate_valid = run.targets['val']['valid'][row:row+1]
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    condition = system.condition(sequence=sequence, inputs=inputs, language=language,
                        candidate_features=run.features['val'][row:row+1], candidate_valid=candidate_valid)
                # Targets do not participate in the conditioning/generation calls above.
                target = teacher.encode_scene({**item['future_frames'], 'token': item['token']},
                                               image_root=run.datasets['val'].image_root)
                if condition['grid'] != target['grid']:
                    raise ValueError('observed/GT grid mismatch')
                trajectory = item['trajectory']
                gt = torch.as_tensor(trajectory['future_traj_10hz'], device='cuda').unsqueeze(0)
                mask = torch.as_tensor(trajectory['future_valid_mask_10hz'], device='cuda').unsqueeze(0)
                generator = torch.Generator(device='cuda').manual_seed(seed * 1000000 + row)
                noise = torch.randn(gt.shape, device='cuda', generator=generator) * run.model.config.noise_init_std
                supervision = {'q': run.targets['val']['q'][row:row+1], 'candidate_valid': candidate_valid,
                    'future_features': target['features'].unsqueeze(0).to('cuda'),
                    'trajectory': gt, 'trajectory_valid': mask}
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    result = system.losses(condition, inputs=inputs, language=language, supervision=supervision,
                        noise=noise, flow_time=torch.full((1,), .5, device='cuda'), weights=weights)
                    sampled = system.sample(condition, inputs=inputs, noise=noise)
                losses = {n: float(v) for n, v in result['losses'].items()}
                if not np.isfinite(list(losses.values())).all():
                    raise ValueError('nonfinite validation loss')
                uniform = torch.zeros_like(condition['reason_scores'])
                uniform_kl = float(kl_divergence(uniform, supervision['q'], candidate_valid).mean())
                metrics = trajectory_metrics(sampled, trajectory['future_traj_10hz'], trajectory['future_valid_mask_10hz'])
                entry = {'token': item['token'], 'log_name': item['log_name'], 'losses': losses,
                    'uniform_kl': uniform_kl, 'metrics': metrics, 'vlm_condition_calls': condition['vlm_calls'],
                    'future_shape': list(condition['future_prediction'].shape), 'reason': reason,
                    'flow_time': .5, 'reason_content_length': len(sequence['reason_positions'])}
                records.append(entry)
                count = condition['future_prediction'].numel()
                future_sse += losses['future'] * count
                future_count += count
                if output is not None:
                    token = item['token']
                    if not token or Path(token).name != token or token in ('.', '..'):
                        raise ValueError('unsafe prediction token')
                    torch.save(condition['future_prediction'].detach().to('cpu', dtype=torch.float16), output / f'{token}.future.pt')
                    (output / f'{token}.json').write_text(json.dumps({**entry,
                        'trajectory': sampled[0].float().cpu().tolist(),
                        'future_file': f'{token}.future.pt', 'future_layout': 'batch,camera,time,row,column,channel'}, indent=2))
                del condition, result, supervision, target, sampled
    finally:
        system.train(was_training)
    return {'kind': 'held_out_joint_evaluation', 'reason_source': reason_source,
        'scene_count': len(records), 'tokens': [r['token'] for r in records], 'seed': seed,
        'endpoint_flow_time': .5, 'planning_metrics_source': 'free_noise_official_sampler',
        'mean_losses': {n: float(np.mean([r['losses'][n] for r in records])) for n in records[0]['losses']},
        'mean_metrics': {n: float(np.mean([r['metrics'][n] for r in records])) for n in records[0]['metrics']},
        'mean_uniform_kl': float(np.mean([r['uniform_kl'] for r in records])),
        'future_mse_element_weighted': future_sse / future_count, 'records': records}


@torch.no_grad()
def predict(run, system, *, limit=1, seed=0, max_new_tokens=256, output_dir):
    """Autoregressive prediction from observable inputs only; no target encoding/loss."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    rows = list(range(len(run.datasets['val'])))
    if limit > 0:
        rows = rows[:limit]
    was_training = system.training
    records = []
    try:
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            system.eval()
            for row in rows:
                item = run.datasets['val'][row]
                inputs = device_inputs(item)
                sequence, language, reason = generate_sequence(run, inputs, max_new_tokens=max_new_tokens)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    condition = system.condition(sequence=sequence, inputs=inputs, language=language)
                    noise = run.model._initial_noise(1, 50, seed + row, inputs['input_ids'].device)
                    noise = noise * run.model.config.noise_init_std
                    sampled = system.sample(condition, inputs=inputs, noise=noise)
                if not torch.isfinite(sampled).all() or not torch.isfinite(condition['future_prediction']).all():
                    raise ValueError('nonfinite joint prediction')
                token = item['token']
                if not token or Path(token).name != token or token in ('.', '..'):
                    raise ValueError('unsafe prediction token')
                torch.save(condition['future_prediction'].detach().to('cpu', dtype=torch.float16), output / f'{token}.future.pt')
                record = {'token': token, 'reason': reason, 'trajectory': sampled[0].float().cpu().tolist(),
                    'future_file': f'{token}.future.pt', 'future_shape': list(condition['future_prediction'].shape),
                    'future_layout': 'batch,camera,time,row,column,channel',
                    'condition_calls': condition['vlm_calls'], 'reason_source': 'generated',
                    'sequence_length': sequence['sequence_length'], 'seed': seed + row}
                (output / f'{token}.json').write_text(json.dumps(record, indent=2))
                records.append(record)
                del condition, sampled
    finally:
        system.train(was_training)
    return {'kind': 'autoregressive_joint_prediction', 'scene_count': len(records),
            'records': records, 'supervision_used': False}
