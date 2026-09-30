from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import nn
from src.model.shared import forward as joint_forward
from src.model.future.branch import NativeFutureBranch, NativeFutureConfig, native_future_mse, observed_future_grid
from src.model.reasoning.scoring import kl_divergence
from src.model.planning.branch import planning_supervision, sample_planning

@dataclass(frozen=True)
class JointLossWeights:
    text: float = 0.
    reason_kl: float = 1.
    future: float = 1.
    planning: float = 1.

class ThreeBranchSystem(nn.Module):

    def __init__(self, model, shared_forward, reason_head, *, future_config=None):
        super().__init__()
        self.model, self.shared_forward, self.reason_head = model, shared_forward, reason_head
        self.future = NativeFutureBranch(future_config or NativeFutureConfig())
        self.model.planning_expert.float().requires_grad_(True)

    def train(self, mode=True):
        super().train(mode)
        self.model.vlm.model.visual.eval()
        return self

    def condition(self, *, sequence, inputs, language, candidate_features=None, candidate_valid=None):
        shared = self.shared_forward(sequence=sequence, inputs=inputs, language=language)
        ids = torch.tensor([sequence['input_ids']],device=inputs['input_ids'].device,dtype=torch.long)
        anchor = self.model._rope_positions(ids,inputs['image_grid_thw'])[:,:,-1]
        grid = observed_future_grid(inputs['image_grid_thw'],merge_size=self.model.processor.merge_size)
        prediction = self.future(shared['scene_kv'],anchor,grid=grid)
        if candidate_features is None:
            if candidate_valid is not None:
                raise ValueError('candidate mask without candidate features')
            probabilities,scores = None,None  # planning inference needs no candidate GT bank
        else:
            probabilities,scores = self.reason_head.probabilities(shared['reason_hidden'],candidate_features,
                                                                  valid=candidate_valid)
        return {**shared,'anchor':anchor,'grid':grid,'future_prediction':prediction,
                'reason_probabilities':probabilities,'reason_scores':scores,
                'planning_kv':shared['scene_kv'],'planning_anchor':anchor}

    def losses(self, condition, *, inputs, language, supervision, noise, flow_time,
               weights=None):
        weights = weights or JointLossWeights()
        if condition['reason_scores'] is None:
            raise ValueError('training KL requires the candidate set')
        device = condition['logits'].device
        targets = torch.as_tensor(language['targets'],device=device,dtype=torch.long)
        text = joint_forward.language_loss(condition['logits'],targets)
        reason = kl_divergence(condition['reason_scores'],supervision['q'],supervision['candidate_valid']).mean()
        future = native_future_mse(condition['future_prediction'],supervision['future_features'])
        planned = planning_supervision(self.model.planning_expert,
            scene_kv=condition['planning_kv'],anchor=condition['planning_anchor'],inputs=inputs,
            trajectory=supervision['trajectory'],valid=supervision['trajectory_valid'],
            scale=self.model.trajectory_scale(device),noise=noise,flow_time=flow_time)
        losses={'text':text,'reason_kl':reason,'future':future,'planning':planned['loss']}
        total=sum(getattr(weights,k)*v for k,v in losses.items())
        return {'total':total,'losses':losses,'planning':planned}

    @torch.no_grad()
    def sample(self, condition, *, inputs, noise):
        with torch.autocast(noise.device.type,dtype=torch.bfloat16,enabled=noise.is_cuda):
            return sample_planning(self.model.planning_expert,
                scene_kv=condition['planning_kv'],anchor=condition['planning_anchor'],inputs=inputs,
                scale=self.model.trajectory_scale(noise.device),noise=noise,
                num_steps=self.model.config.num_inference_steps,min_one_minus_t=self.model.config.min_one_minus_t)

    def parameter_groups(self):
        planner=list(self.model.planning_expert.parameters())
        planner_ids={id(p) for p in planner}
        return {'vlm_lora':[p for p in self.model.parameters() if p.requires_grad and id(p) not in planner_ids],
                'reason_head':list(self.reason_head.parameters()),'future':list(self.future.parameters()),
                'planner':planner}
