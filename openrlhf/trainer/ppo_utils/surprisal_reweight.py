"""
Surprisal-guided token-level advantage reweighting.

Adapted from STARE (Surprisal-Guided Token-Level Advantage Reweighting for
Policy Entropy Stability, https://arxiv.org/abs/2606.19236). GRPO-style RLVR
training tends to suffer from policy entropy collapse: a first-order analysis of
the per-token entropy dynamics shows that the entropy variation of a token
decomposes into the product of its trajectory-level advantage and an entropy
sensitivity that grows with the token's surprisal (-log pi(a|s)). High-surprisal
tokens are therefore "entropy-critical" -- they dominate how fast the policy
loses (or keeps) its exploration.

This module implements the core, stateless slice of that idea: identify the
entropy-critical token subset via *batch-internal surprisal quantiles* and
selectively reweight the effective advantage of those tokens. Boosting the
advantage magnitude of high-surprisal tokens keeps gradient pressure on the
exploratory part of the distribution and slows entropy collapse, without
changing the PolicyLoss contract (it still consumes a per-token advantage tensor
plus an action mask).

Intentionally out of scope (not needed for this value): STARE's full
target-entropy closed-loop gate, which requires the complete next-token
distribution and cross-step entropy tracking. Here we operate purely on the
chosen-token log-probs already carried on each Experience for one batch.
"""

from typing import List, Optional

import torch

from openrlhf.utils.logging_utils import init_logger

logger = init_logger(__name__)


def _surprisal_threshold(surprisals: torch.Tensor, quantile: float) -> torch.Tensor:
    """Return the batch-internal surprisal value at the given quantile.

    Uses ``kthvalue`` rather than ``torch.quantile`` so the computation stays
    valid for the very large flattened token counts seen in real RL batches
    (``torch.quantile`` rejects inputs above ~16M elements).
    """
    n = surprisals.numel()
    # k is 1-indexed for kthvalue; clamp into [1, n].
    k = int(quantile * n) + 1
    k = max(1, min(k, n))
    return torch.kthvalue(surprisals, k).values


def compute_surprisal_reweight_mask(
    advantages: torch.Tensor,
    action_log_probs: torch.Tensor,
    action_mask: Optional[torch.Tensor],
    threshold: torch.Tensor,
    factor: float,
    sign_aware: bool = True,
) -> torch.Tensor:
    """Reweight the advantages of entropy-critical tokens for one experience.

    Args:
        advantages: (B, A) per-token advantages.
        action_log_probs: (B, A) log pi(a|s) of the sampled tokens.
        action_mask: (B, A) bool/0-1 mask over response tokens, or ``None``.
        threshold: scalar surprisal value; tokens at/above it are entropy-critical.
        factor: multiplier applied to the entropy-critical tokens' advantages.
        sign_aware: if True (the STARE four-quadrant view), only amplify
            exploration-promoting tokens (positive advantage on a high-surprisal
            token reinforces a low-probability choice and raises entropy). Tokens
            with negative advantage are left unchanged so the reweighting cannot
            accelerate collapse.

    Returns:
        ``(reweighted_advantages, critical_mask)`` where ``critical_mask`` is the
        boolean tensor of tokens that were treated as entropy-critical.
    """
    surprisal = -action_log_probs.float()
    critical = surprisal >= threshold
    if action_mask is not None:
        critical = critical & action_mask.bool()
    if sign_aware:
        critical = critical & (advantages > 0)

    weight = torch.ones_like(advantages, dtype=advantages.dtype)
    weight = torch.where(critical, weight * factor, weight)
    return advantages * weight, critical


def apply_surprisal_reweighting(experiences: List, args) -> None:
    """Apply STARE surprisal-guided advantage reweighting to a batch in place.

    Gated by ``args.algo.advantage.surprisal_reweight`` (default off). The
    surprisal quantile threshold is computed once across *all* valid tokens in
    the batch (batch-internal quantiles), then each experience's advantages are
    reweighted against that shared threshold. Per-batch diagnostics are written
    to ``experience.info`` for logging.

    Args:
        experiences: list of Experience objects with ``advantages``,
            ``action_log_probs`` and ``action_mask`` already populated.
        args: training arguments namespace.
    """
    advantage_cfg = args.algo.advantage
    if not getattr(advantage_cfg, "surprisal_reweight", False):
        return

    quantile = float(getattr(advantage_cfg, "surprisal_reweight_quantile", 0.8))
    factor = float(getattr(advantage_cfg, "surprisal_reweight_factor", 1.5))
    sign_aware = bool(getattr(advantage_cfg, "surprisal_reweight_sign_aware", True))

    usable = [
        exp
        for exp in experiences
        if getattr(exp, "advantages", None) is not None and getattr(exp, "action_log_probs", None) is not None
    ]
    if not usable:
        return

    # ── Collect masked surprisals across the whole batch for the quantile ──
    masked_surprisals = []
    for exp in usable:
        surprisal = -exp.action_log_probs.float().flatten()
        if exp.action_mask is not None:
            mask = exp.action_mask.bool().flatten()
            surprisal = surprisal[mask]
        if surprisal.numel() > 0:
            masked_surprisals.append(surprisal)

    if not masked_surprisals:
        return

    all_surprisals = torch.cat(masked_surprisals)
    threshold = _surprisal_threshold(all_surprisals, quantile)

    total_critical = 0
    total_tokens = 0
    for exp in usable:
        exp.advantages, critical = compute_surprisal_reweight_mask(
            exp.advantages,
            exp.action_log_probs,
            exp.action_mask,
            threshold=threshold.to(exp.advantages.device),
            factor=factor,
            sign_aware=sign_aware,
        )
        num_critical = int(critical.sum().item())
        num_tokens = int(exp.action_mask.bool().sum().item()) if exp.action_mask is not None else critical.numel()
        total_critical += num_critical
        total_tokens += num_tokens
        if "reward" in exp.info:  # only attach when this batch is being logged
            exp.info["surprisal_critical_frac"] = torch.tensor(
                num_critical / max(num_tokens, 1), device=exp.rewards.device
            ).repeat(len(exp.rewards))

    logger.info(
        f"[STARE Surprisal Reweight] threshold={threshold.item():.4f}, factor={factor}, "
        f"quantile={quantile}, sign_aware={sign_aware}, "
        f"{total_critical}/{total_tokens} tokens reweighted"
    )
