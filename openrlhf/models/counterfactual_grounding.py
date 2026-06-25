"""Cross-modal counterfactual grounding regularizer for VLM RL.

Adapted from CounterFactual Policy Optimization (CFPO),
https://arxiv.org/abs/2606.23206. CFPO enforces causal consistency between
visual perception and textual reasoning by regularizing the policy with the
discrepancy between its factual prediction and a *counterfactual* prediction
made after critical visual cues are suppressed. Maximizing that discrepancy
discourages the model from ignoring visual evidence in favor of language
priors, which is the root cause of grounding failures and hallucination
drift during long chain-of-thought reasoning.

This module provides the two ingredients CFPO needs on top of an existing
GRPO/DAPO-style policy loss:

* :func:`suppress_visual_inputs` builds the counterfactual multimodal inputs
  by zeroing the visual content tensors while keeping their shape (so the
  image-placeholder token bookkeeping the model relies on stays valid).
* :func:`counterfactual_grounding_loss` measures the per-token discrepancy
  between the factual and counterfactual action log-probs and aggregates it
  into the scalar ``kl_cmve`` term to be *maximized*.

It deliberately stays a regularizer that plugs into the existing actor
training step: no external reward model or extra supervision is required,
matching the paper's "seamlessly integrates with GRPO/DAPO" claim.
"""

from typing import Dict, Optional

import torch

from .loss import aggregate_loss
from .utils import compute_approx_kl

# Tensors that carry the actual pixel content of the multimodal inputs.
# Grid / thw metadata is intentionally left untouched so the suppressed
# forward still produces the same number of visual placeholder embeddings.
_VISUAL_CONTENT_KEYS = ("pixel_values", "pixel_values_videos")


def suppress_visual_inputs(mm_inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Return counterfactual multimodal inputs with visual cues suppressed.

    The visual content tensors (``pixel_values`` / ``pixel_values_videos``)
    are replaced with zeros of the same shape, which removes the visual
    signal while preserving the token-count contract the model assumes
    between image placeholders and visual features. All other entries
    (grid sizes, attention masks, ...) are shared by reference, since they
    are not mutated by the forward pass.

    Returns an empty dict when ``mm_inputs`` carries no visual content, which
    the caller uses to skip the counterfactual forward entirely.
    """
    if not mm_inputs:
        return {}

    has_visual = any(key in mm_inputs for key in _VISUAL_CONTENT_KEYS)
    if not has_visual:
        return {}

    counterfactual = dict(mm_inputs)
    for key in _VISUAL_CONTENT_KEYS:
        tensor = mm_inputs.get(key)
        if tensor is not None:
            counterfactual[key] = torch.zeros_like(tensor)
    return counterfactual


def counterfactual_grounding_loss(
    action_log_probs: torch.Tensor,
    counterfactual_log_probs: torch.Tensor,
    action_mask: Optional[torch.Tensor] = None,
    kl_estimator: str = "k3",
    token_level_loss: bool = True,
    dp_size: int = 1,
    batch_num_tokens: Optional[float] = None,
    global_batch_size: Optional[float] = None,
) -> torch.Tensor:
    """Cross-modal counterfactual enhancement discrepancy (CFPO ``kl_cmve``).

    Measures the per-token divergence between the factual action log-probs and
    the counterfactual ones (obtained from a forward with visual cues
    suppressed), then aggregates it with the same reduction the policy loss
    uses. A non-negative KL estimator (``k3`` by default) keeps the result
    ``>= 0`` so it can be treated as a discrepancy to *maximize*: the caller
    subtracts ``coef * discrepancy`` from the training loss.

    A larger value means the model's predictions depend more strongly on the
    visual evidence, i.e. better grounding.
    """
    discrepancy = compute_approx_kl(
        action_log_probs,
        counterfactual_log_probs,
        kl_estimator=kl_estimator,
    )
    return aggregate_loss(
        discrepancy,
        action_mask,
        token_level_loss=token_level_loss,
        dp_size=dp_size,
        batch_num_tokens=batch_num_tokens,
        global_batch_size=global_batch_size,
    )
