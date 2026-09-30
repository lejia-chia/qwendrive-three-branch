"""Model/data assembly for the current joint baseline only."""
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from src.utils.paths import ROOT
from src.data.q_targets import DEFAULT_Q_ARTIFACT,load_q_artifact
from src.data.reason_text import ReasonTextStore
from src.data.dataset import NativeNavtrainDataset
from src.model.shared import forward as joint_forward
from src.model.shared.kv_export import set_checkpointing
from src.model.reasoning.scoring import ReasonScoreHead,all_features
from qwen_drive import QwenDriveForPlanning

class JointRuntime:
    def __init__(self,args):
        torch.set_num_threads(4)
        torch.set_float32_matmul_precision('high')
        self.q_root=Path(args.q_artifact).resolve()
        self.q={s:load_q_artifact(self.q_root,s) for s in ('train','val')}
        self.model=QwenDriveForPlanning.from_pretrained(str(ROOT/'model'),dtype=torch.bfloat16,
            attn_implementation='flash_attention_2').to('cuda').eval()
        # Reject retired special-token ids without adding them to the tokenizer.
        self.t_pred={s:ReasonTextStore(s) for s in ('train','val')}
        processor=self.model.processor
        self.table=SimpleNamespace(
            forbidden_ids=self.t_pred['train'].manifest['forbidden_token_ids'],
            im_end_id=processor.im_end_id, newline_ids=list(processor.newline_ids),
            terminator_ids=(processor.im_end_id, self.model.config.vlm_config.text_config.eos_token_id))
        self.lora=joint_forward.freeze_and_apply_lora(self.model,seed=args.seed)
        torch.manual_seed(args.seed)
        self.head=ReasonScoreHead().to('cuda')
        self.forward=joint_forward.JointForward(self.model)
        set_checkpointing(self.model,enabled=True,uncheckpointed_layers=self.forward.full_layers)
        self.model.train()
        self.datasets,self.features,self.targets={},{},{}
        for split in ('train','val'):
            arrays=self.q[split]['arrays'];tokens=[str(t) for t in arrays['tokens']]
            self.datasets[split]=NativeNavtrainDataset(ROOT/'data/native',self.model.processor,
                tokens=tokens,split=split,expect_version='current')
            if set(tokens)!=set(self.t_pred[split].rows):
                raise ValueError(f'{split}: q and reason-text tokens differ')
            self.features[split]=torch.as_tensor(all_features(arrays),device='cuda')
            self.targets[split]={'q':torch.as_tensor(arrays['q'],dtype=torch.float32,device='cuda'),
                'valid':torch.as_tensor(arrays['valid']>0,device='cuda'),'gt_index':arrays['gt_indices']}
        self.train_tokens=self.datasets['train'].tokens
        self.train_rows=np.flatnonzero(self.q['train']['arrays']['record_status']==0).tolist()
        if not self.train_rows:raise ValueError('no eligible training scene')
        self._train_item_cache={}
    def train_item(self,position):
        if position not in self._train_item_cache:
            self._train_item_cache[position]=self.datasets['train'][position]
        return self._train_item_cache[position]
