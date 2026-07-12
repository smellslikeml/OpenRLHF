"""Entropy-adaptive per-token filtering via the Relative Surprisal Index (RSI).

Adapted from "Which Tokens Matter? Adaptive Token Selection for RLVR with the
Relative Surprisal Index" (arXiv:2606.31575), which couples a position's
predictive entropy with the probability the policy assigned to the *sampled*
token. RSI Selection (RSI-S) keeps only tokens whose RSI lies in a stable
interval, dropping both redundant low-surprisal tokens (the policy is already
confident there) and unstable high-surprisal tail tokens (rare under the
policy). In OpenRLHF it is applied as a per-token policy-loss filter in
``PolicyLoss.forward``, next to the existing tis/icepop interval family.

RSI is the sampled token's surprisal relative to the predictive entropy::

    RSI = (-log p_sel) / H

where ``p_sel`` is the policy probability of the sampled token and ``H`` is the
predictive entropy of the next-token distribution (both in nats). ``RSI ~= 1``
means the sampled token is "typical" -- its surprisal matches the expected
surprisal. ``RSI << 1`` flags a redundant near-deterministic token; ``RSI >> 1``
flags an unstable tail token. The metric is parameter-free: both signals are
already computed in the actor forward pass.
"""

from typing import Tuple

import torch


def relative_surprisal_index(
    entropy: torch.Tensor,
    selected_log_probs: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Per-token Relative Surprisal Index = surprisal / predictive entropy.

    Args:
        entropy: Per-token predictive entropy ``H`` of the policy distribution
            (nats), shape ``[batch, seq]``.
        selected_log_probs: Per-token log-probability the policy assigned to the
            sampled token (``log p_sel``), same shape as ``entropy``.
        eps: Numerical floor for the entropy denominator so deterministic
            positions (``H -> 0``) yield ``RSI -> 0`` (redundant) rather than
            dividing by zero.

    Returns:
        Per-token RSI, detached from the autograd graph. The selection is a
        fixed per-step schedule, not a learned quantity, so it must not carry
        gradient into the policy loss it gates.
    """
    surprisal = -selected_log_probs
    return (surprisal / (entropy + eps)).detach()


def rsi_select_mask(
    entropy: torch.Tensor,
    selected_log_probs: torch.Tensor,
    low: float,
    high: float,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Tokens whose RSI lies in the stable interval ``[low, high]``.

    Tokens outside the interval are dropped from the policy loss -- redundant
    low-surprisal tokens (``RSI < low``) on one side and unstable high-surprisal
    tail tokens (``RSI > high``) on the other.

    Returns:
        ``(mask, rsi)`` where ``mask`` is a boolean tensor (True = keep) and
        ``rsi`` is the detached per-token index, both shaped like ``entropy``.
        ``rsi`` is returned so callers can log what fraction of tokens survived.
    """
    rsi = relative_surprisal_index(entropy, selected_log_probs, eps=eps)
    mask = (rsi >= low) & (rsi <= high)
    return mask, rsi
