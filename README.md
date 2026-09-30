# QwenDrive: Reason / Future / Planning

Current baseline: one teacher-forced `S + T_text + im_end + newline` VLM forward; three supervised branches share its differentiable representations. The official QwenDrive source and released backbone / RL planner weights initialize the model.

```text
Qwendrive/
├── train.py
├── scripts/train.sh
├── src/
│   ├── model/
│   │   ├── shared/      # full-attention LoRA, differentiable forward, KV export
│   │   ├── reasoning/   # fixed candidate features, scoring head, masked KL
│   │   ├── future/      # native-grid prediction, frozen visual target encoder
│   │   ├── planning/    # official full PlanningExpert, flow endpoint supervision
│   │   └── joint.py     # three-branch assembly and interfaces
│   ├── data/            # native scenes, text pseudo-labels, verified q targets
│   ├── training/        # runtime, trainer, safe checkpoint/resume
│   └── utils/           # paths and streaming fingerprints
├── data/
│   ├── scenes/          # observable scene records + trajectory GT
│   ├── native/          # full navtrain split indices + future frame paths
│   ├── reason_text/     # current natural-language pseudo-labels only
│   └── q_targets/v2/    # current candidate sets and distance-aware distributions
├── model/               # released backbone + planner-rl
├── Qwen-Drive-1.0/       # unchanged upstream source, including its license
├── outputs/             # new runs only
├── requirements.txt
├── environment.lock.txt
└── docs/architecture.md
```

## Data and supervision

`S` is the official processor output: 12 observable images (3 cameras × 4 historical/current frames), navigation text and history information in the prompt. Explicit history / ego tensors also enter PlanningExpert. `T_text` is a frozen generator's natural-language pseudo-label, copied without behaviour tokens. GT trajectory and future images never enter the VLM prompt.

| Branch | Input | Output | Supervision |
|---|---|---|---|
| Reason | Reason hidden from `S+T`, fixed features for each candidate | candidate scores / probabilities | text CE + `KL(q || p)` |
| Future | graph-connected `S+T` K/V and observed image grid | `[B,3,10,H,W,2560]` | MSE against frozen Qwen visual features of future GT frames |
| Planning | the **same** `S+T` K/V, anchor, explicit ego/history; flow-noised waypoints during training | 50 × `(x,y,yaw)` | masked normalized clean-endpoint MSE with periodic yaw residual |

Future predicts at +0.5, +1.0, ..., +5.0 seconds. Native grid is measured from images (the checked scene is 22×40); there is no additional spatial pooling or image decoder.

Future predictions are **not** given to Planning. Both losses backpropagate through the shared K/V into VLM LoRA. K/V are activations computed again at each forward, not optimizer-owned parameters. Planning loss does not update Future-head parameters directly.

Trainable components: VLM full-attention LoRA, Reason scoring head, native Future head and the **entire** PlanningExpert. Backbone and visual teacher remain frozen. The total loss is:

`lambda_text L_text + lambda_kl L_reason + lambda_future L_future + lambda_planning L_planning`. The default `lambda_text=0` disables text CE supervision; the other three weights default to 1. Frozen `T_text` still conditions the shared forward. CE is still computed for reporting, so setting its weight to zero does not remove its compute cost.

The underlying native split contains **90,975 train / 11,160 val**, separated by logs. The **currently prepared joint pool is 598 train / 99 diagnostic val** because text and candidate targets cover that subset. This entry does not imply full-navtrain or eight-GPU support. The current training loop saves checkpoints, training losses and held-out joint metrics. NAVSIM PDMS is not included.

## Run

The existing `qwendrive` environment is usable. Python 3.12 / CUDA 12.8 are required; `requirements.txt` lists runtime versions, while `environment.lock.txt` records the actual environment. Build flash-attn / causal-conv1d against the installed CUDA PyTorch with `--no-build-isolation`. This repository expects the upstream source at its shown location and released weights under `model/`.

```bash
cd /home/tione/code/jiale_v/Experiments/Qwendrive

# Train the current joint pool; no separate smoke stage is required.
./scripts/train.sh --mode train --steps 150 --checkpoint-every 50 \
  --seed 0 --lr 1e-4 --planner-lr 1e-5 \
  --output-dir outputs/joint_seed0

# Resume into a new output directory; model/data/protocol must match.
./scripts/train.sh --mode train --steps 300 --seed 0 \
  --resume outputs/joint_seed0/checkpoints/step_000150.pt \
  --output-dir outputs/joint_seed0_resume
```

`PYTHON=/path/to/python ./scripts/train.sh ...` overrides the interpreter. Existing output directories are rejected. `--mode check` performs one real sample's gradient routing and optimizer-step check without starting a training run.

Scene record paths in the indices are relative to the workspace. The external OpenScene image root is recorded in `data/native/manifest.json`; a new machine must mount that image source or update the image-root setting with corresponding provenance. Original OpenScene files are outside this repository and were not deleted. Historical source hashes in the q-target manifest describe its construction; those historical files are not runtime dependencies.

## Limits

Planning supervision is an explicit supervised flow endpoint baseline, not a reconstruction of unpublished official RL training. Optimizer-owned PlanningExpert parameters use FP32 with BF16 autocast; initial outputs are not promised to be bit-identical to the sealed all-BF16 inference run. Teacher-forced pseudo-label training and autoregressive reasoning inference remain different modes. No prediction-quality or planning-improvement claim follows from the engineering checks.

## Corrected execution contract

LoRA A/B and AdamW moments are FP32; BF16 autocast handles compute. Text CE
supervises natural-language content and `im_end`. Reason pooling excludes closing
format tokens. Future and Planning both attend to the same closed assistant turn.
Candidate features ignore all masked coordinates, including cumulative arc length.

Every saved training step evaluates a fixed eligible val subset (default 10 scenes;
`--val-scenes 0` selects all 99). Reports separate text CE, Reason KL, native Future
MSE and denoising endpoint MSE at t=0.5. XY ADE/FDE in metres come from free-noise
PlanningExpert sampling, with yaw error separately in radians. `ok` indicates
successful execution; it does not claim improved planning quality.

```bash
./scripts/train.sh --mode train --steps 150 --checkpoint-every 50 \
  --val-scenes 10 --generated-val --output-dir outputs/joint_seed0

./scripts/train.sh --mode infer \
  --resume outputs/joint_seed0/checkpoints/step_000150.pt \
  --val-scenes 1 --output-dir outputs/joint_prediction
```

`--generated-val` also evaluates Reason generated from S, separately from frozen
pseudo-label conditioning. Inference generates T from observable inputs, then runs
one closed S+T forward for shared K/V and predicts Future/Planning. Outputs contain
Reason text, a 50x3 trajectory and an FP16 native feature tensor (not decoded video).
Generation reaching the token limit without a terminator is rejected. Inference
does not encode future GT images or read q/GT as prediction conditioning.

Checkpoint format is `qwendrive_three_branch_shared_kv_v3`; the previous open-turn,
BF16-LoRA training contract is rejected. Resume uses the exact recorded protocol;
inference checks model/data/code identity while permitting a different seed/subset.

Current fix verification is recorded in [docs/verification.json](docs/verification.json):
31 CPU integration checks, 24 targeted regression checks, 20 real GPU checks,
and actual checkpoint save/resume plus independent autoregressive inference.
Temporary verification checkpoints and predictions are removed after acceptance.
