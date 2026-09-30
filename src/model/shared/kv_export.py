"""Explicit post-RoPE K/V export for the shared-forward plan.

Why an interface instead of a hook
----------------------------------
``Qwen3_5Attention.forward`` returns ``(attn_output, attn_weights)``: the post-RoPE
key/value tensors are local variables, and the only place they surface is
``past_key_values.update(key_states, value_states, self.layer_idx)``.  A forward
hook on the attention module therefore cannot see them.  Worse,
``GradientCheckpointingLayer.__call__`` *forces* ``past_key_values=None`` while a
layer is checkpointed in training mode, so under checkpointing the cache route is
closed by construction (that is exactly the incompatibility the review flagged —
it is not an assumption, it is in the code path).

``ExportCache`` piggybacks on the cache object the attention module already calls
and stores exactly the tensors the attention ends up using: whatever
``DynamicCache.update`` returns, i.e. the graph-connected post-RoPE K/V.  Layers
whose K/V must carry the graph into a second loss head therefore have to run
un-checkpointed; the probe measures what that costs in memory instead of guessing.
"""
from __future__ import annotations

import torch
from transformers.cache_utils import DynamicCache


def text_layers(model):
    """The 32 decoder layers of the text backbone."""
    return model.vlm.model.language_model.layers


def full_attention_layer_indices(model) -> tuple[int, ...]:
    """Absolute indices of the full-attention layers, from the model's own config.

    Cross-checked against each layer's ``block_type`` so a config that disagrees
    with the constructed layers is refused rather than silently mis-exported.
    """
    declared = tuple(int(index) for index in model.config.full_attention_layers)
    actual = tuple(index for index, layer in enumerate(text_layers(model))
                   if getattr(layer, "block_type", None) == "full_attention")
    if declared != actual:
        raise SystemExit(f"config.full_attention_layers {declared} != constructed "
                         f"full-attention layers {actual}")
    return declared


class ExportCache(DynamicCache):
    """A DynamicCache that additionally keeps the K/V of the wanted layers.

    Contract: **one ExportCache per sample / per forward**.  This is a training
    cache for a single full-sequence forward, not a generation cache — the base
    cache is never cleared here, so reusing the object for a second forward would
    both append to the previous sequence and mix the two autograd graphs.
    ``update`` refuses a second entry for the same layer so that reuse fails
    loudly instead of silently conditioning the next sample on the previous one.

    ``update`` records the tensors it returns — the same objects attention then
    attends over — so they stay attached to the autograd graph.
    """

    def __init__(self, wanted: tuple[int, ...], config=None):
        # the hybrid backbone's linear-attention layers call update_conv_state /
        # update_recurrent_state on the SAME cache object, so the per-layer
        # structure has to be built from the model config exactly as the model
        # itself builds it (DynamicCache(config=...) in Qwen3_5TextModel.forward)
        super().__init__(config=config)
        self.wanted = set(int(index) for index in wanted)
        self.exported: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

    def update(self, key_states, value_states, layer_idx=None, *args, **kwargs):
        if layer_idx is not None and int(layer_idx) in self.wanted \
                and int(layer_idx) in self.exported:
            raise RuntimeError(
                f"layer {layer_idx} was already exported by this cache: an ExportCache "
                "serves exactly one forward — create a new one per sample instead of "
                "reusing it (the base cache is not cleared, so a second forward would "
                "append to the previous sequence and mix the two graphs)")
        keys, values = super().update(key_states, value_states, layer_idx, *args, **kwargs)
        if layer_idx is not None and int(layer_idx) in self.wanted:
            self.exported[int(layer_idx)] = (keys, values)
        return keys, values




def set_checkpointing(model, *, enabled: bool,
                      uncheckpointed_layers: tuple[int, ...] = ()) -> dict:
    """Turn checkpointing on/off, then exempt the listed layers.

    An exempted layer keeps its activations (and therefore its K/V) on the graph,
    which is what lets a second loss head backpropagate through them.  Returns the
    per-layer flags actually in force.
    """
    if enabled:
        model.vlm.gradient_checkpointing_enable()
    else:
        model.vlm.gradient_checkpointing_disable()
    layers = text_layers(model)
    for index in uncheckpointed_layers:
        layers[index].gradient_checkpointing = False
    return {str(index): bool(layer.gradient_checkpointing)
            for index, layer in enumerate(layers)}
