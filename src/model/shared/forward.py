from __future__ import annotations

import torch
import torch.nn.functional as F
from src.model.shared import lora as branch
from src.data import reason_text as t_pred_module
from src.model.shared.kv_export import ExportCache,full_attention_layer_indices,text_layers

CONTRACT_VERSION = "v2-text-closed-turn"


def freeze_and_apply_lora(model, *, seed: int = 0) -> dict:
    """Freeze the complete QwenDrive model, then expose only the LoRA weights."""
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    installed = branch.apply_lora(model, seed=seed)
    allowed = {id(parameter)
               for module in model.modules() if isinstance(module, branch.LoRALinear)
               for parameter in (module.lora_a, module.lora_b)}
    trainable = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    if not allowed or trainable != allowed:
        raise SystemExit("joint VLM trainable parameters differ from the LoRA-only allowlist")
    return {**installed, "trainable_tensors": len(trainable),
            "trainable_params": sum(parameter.numel() for parameter in model.parameters()
                                    if parameter.requires_grad)}


def build_sequence(prompt_ids: list[int], record: dict, table: object,
                   *, expected_token: str) -> dict:
    """Bind a frozen text prediction to its original scene before forming S + T."""
    if record.get("token") != expected_token:
        raise SystemExit(f"T_text token {record.get('token')} != scene token {expected_token}")
    if (len(prompt_ids) != record.get("prompt_length") or
            t_pred_module.prompt_ids_sha256(prompt_ids) != record.get("prompt_ids_sha256")):
        raise SystemExit(f"{expected_token}: current scene prompt differs from T_text conditioning")
    quality = record.get("quality", {})
    structural_flags = quality.get("structural_flags")
    if not isinstance(structural_flags, list) or structural_flags:
        raise SystemExit(f"{expected_token}: unusable T_text structural flags {structural_flags}")
    t_text_ids = record.get("text_ids")
    if not prompt_ids or not isinstance(t_text_ids, list) or not t_text_ids:
        raise SystemExit("build_sequence: empty prompt or empty T_text")
    if any(not isinstance(token, int) for token in t_text_ids):
        raise SystemExit(f"{expected_token}: T_text contains non-integer token ids")
    terminators = set(table.terminator_ids)
    if any(token in terminators for token in t_text_ids):
        raise ValueError("T_text must contain content only, without a turn terminator")
    return content_sequence(prompt_ids, t_text_ids, table)


def content_sequence(prompt_ids, t_text_ids, table):
    """Canonical closed assistant turn; Reason hidden covers content only."""
    if not prompt_ids or not t_text_ids:
        raise ValueError("empty prompt/reason content")
    closing = [int(table.im_end_id)] + list(table.newline_ids)
    if not closing or any(type(t) is not int or t < 0 for t in closing + list(t_text_ids)):
        raise ValueError("invalid content/closing token IDs")
    if any(t in table.terminator_ids for t in t_text_ids):
        raise ValueError("reason content contains a turn terminator")
    sequence = list(prompt_ids) + list(t_text_ids) + closing
    start = len(prompt_ids)
    end = start + len(t_text_ids)
    present = sorted(set(sequence) & set(table.forbidden_ids))
    if present:
        raise SystemExit(f"the joint sequence contains retired behaviour tokens {present}; "
                         "the v1 contract keeps them out of the input entirely")
    return {"input_ids": sequence, "reason_span": (start, end),
            "reason_positions": list(range(start, end)),
            "prompt_length": start, "sequence_length": len(sequence),
            "contract": CONTRACT_VERSION, "turn_end_span": (end, len(sequence)),
            "language_end": end + 1}


def build_language_targets(sequence: dict) -> dict:
    """Shifted targets over content + im_end; format newline remains conditioning."""
    start, end = sequence["reason_span"]
    if start < 1 or end > len(sequence["input_ids"]) or end <= start:
        raise SystemExit("build_language_targets: invalid reason span")
    language_end = sequence["language_end"]  # content + im_end; fixed newline is conditioning
    if language_end != end + 1 or language_end > len(sequence["input_ids"]):
        raise ValueError("invalid language closing span")
    keep = list(range(start - 1, language_end - 1))
    return {"keep": keep, "targets": list(sequence["input_ids"][start:language_end])}


def language_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits.float(), targets)


class JointForward:
    """One main forward, three exits, all differentiable.

    ``hidden_layer`` defaults to the last decoder layer; it must not be
    checkpointed (see the module docstring) and the class checks the layer flags
    before capturing anything.
    """

    def __init__(self, model, *, hidden_layer: int | None = None):
        allowed = {id(parameter)
                   for module in model.modules() if isinstance(module, branch.LoRALinear)
                   for parameter in (module.lora_a, module.lora_b)}
        trainable = {id(parameter) for parameter in model.parameters()
                     if parameter.requires_grad}
        if not allowed or trainable != allowed:
            raise SystemExit("JointForward requires a frozen backbone and exactly the LoRA "
                             "A/B tensors trainable; call freeze_and_apply_lora first")
        self.model = model
        self.full_layers = full_attention_layer_indices(model)
        layers = text_layers(model)
        self.hidden_layer = len(layers) - 1 if hidden_layer is None else hidden_layer
        if not 0 <= self.hidden_layer < len(layers):
            raise SystemExit(f"hidden_layer {self.hidden_layer} is out of range")

    # ------------------------------------------------------------------ guards

    def _check_hidden_layer(self) -> None:
        layer = text_layers(self.model)[self.hidden_layer]
        if getattr(layer, "gradient_checkpointing", False) and layer.training:
            raise SystemExit(
                f"layer {self.hidden_layer} is checkpointed while training: its output comes "
                "from a discarded pass, so the captured hidden states would carry a stale "
                "graph.  Exempt this layer from checkpointing (see kv_export.set_checkpointing) "
                "or capture a different layer.")

    # ------------------------------------------------------------------ forward

    def __call__(self, *, sequence: dict, inputs: dict, language: dict,
                 capture_hidden: bool = True) -> dict:
        self._check_hidden_layer()
        model = self.model
        if sequence.get("contract") != CONTRACT_VERSION:
            raise SystemExit("the joint input has a different contract version")
        if language != build_language_targets(sequence):
            raise SystemExit("language targets or shifted positions disagree with the joint input")
        prompt = inputs["input_ids"]
        if prompt.ndim != 2 or prompt.shape[0] != 1 or \
                prompt[0].tolist() != sequence["input_ids"][:sequence["prompt_length"]]:
            raise SystemExit("the VLM visual inputs and joint text use different scene prompts")
        device = next(model.parameters()).device
        input_ids = torch.tensor([sequence["input_ids"]], dtype=torch.long, device=device)
        modality = branch.modality_ids(model, input_ids)
        keep = torch.tensor(language["keep"], dtype=torch.long, device=device)
        cache = ExportCache(self.full_layers, config=model.vlm.config)
        captured: dict = {}
        handle = None
        if capture_hidden:
            start, end = sequence["reason_span"]

            def hook(_module, _args, output):
                hidden = output[0] if isinstance(output, tuple) else output
                captured["hidden"] = hidden[:, start:end]

            handle = text_layers(model)[self.hidden_layer].register_forward_hook(hook)
        calls = {"vlm": 0}

        def count(_module, _args):
            calls["vlm"] += 1

        counter = model.vlm.register_forward_pre_hook(count)
        try:
            outputs = model.vlm(
                input_ids=input_ids,
                pixel_values=inputs["pixel_values"],
                image_grid_thw=inputs["image_grid_thw"],
                mm_token_type_ids=modality,
                past_key_values=cache,
                use_cache=True,
                logits_to_keep=keep)
        finally:
            counter.remove()
            if handle is not None:
                handle.remove()
        logits = outputs.logits[0]
        if logits.shape[0] != len(language["keep"]):
            raise SystemExit(f"logits_to_keep returned {logits.shape[0]} positions, "
                             f"expected {len(language['keep'])}")
        scene_kv = []
        for index in self.full_layers:
            if index not in cache.exported:
                raise SystemExit(f"no exported K/V for layer {index}: it must not be "
                                 "checkpointed while the Future exit is used")
            keys, values = cache.exported[index]
            # the cache layout is [batch, kv_heads, sequence, head_dim]; the transpose
            # into the consumer layout [batch, sequence, kv_heads, head_dim] happens below
            if keys.shape[2] != sequence["sequence_length"]:
                raise SystemExit(f"layer {index}: K/V covers {keys.shape[2]} tokens but the "
                                 f"sequence has {sequence['sequence_length']}")
            scene_kv.append((keys.transpose(1, 2), values.transpose(1, 2)))
        if capture_hidden and "hidden" not in captured:
            raise SystemExit("hidden states were not captured")
        result = {"logits": logits, "keep": list(language["keep"]),
                  "scene_kv": scene_kv, "vlm_calls": calls["vlm"],
                  "sequence_length": sequence["sequence_length"],
                  "reason_span": sequential_span(sequence["reason_span"]),
                  "full_attention_layers": list(self.full_layers)}
        if capture_hidden:
            result["reason_hidden"] = captured["hidden"]
        return result


def sequential_span(span) -> list[int]:
    return [int(span[0]), int(span[1])]
