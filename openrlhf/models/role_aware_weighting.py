"""Role-aware per-token policy-loss weighting.

Adapted from "Improving General Role-Playing Agents via Psychology-Grounded
Reasoning and Role-Aware Policy Optimization" (RAPO), arXiv:2606.27025.

RAPO observes that under a learned reward model, generic reward-hacking
phrases and genuinely role-specific phrases receive *identical* policy
gradients, so reward hacking accumulates over training. It corrects this by
weighting gradients with profile--token mutual information: amplify
role-specific tokens when the advantage is positive and attenuate them when
it is negative.

This module delivers that core result as a per-token loss weight that
multiplies into the standard PPO policy loss, mirroring OpenRLHF's existing
token-level importance-sampling correction (``PolicyLoss``). The *profile*
is taken to be the prompt region of each trajectory, and a response token is
considered role-specific when it overlaps that prompt's vocabulary -- a
parameter-free mutual-information proxy that needs no extra checkpoints,
datasets, or reward models. Only the asymmetric gradient-weighting idea is
ported; the paper's Psy-CoT reasoning template and full RAPO training
procedure are intentionally out of scope here.
"""

from typing import Optional

import torch


def profile_overlap_specificity(
    sequences: torch.Tensor,
    action_mask: torch.Tensor,
) -> torch.Tensor:
    """Per-token role-specificity as prompt/response vocabulary overlap.

    Args:
        sequences: ``(B, T)`` token ids covering ``[prompt + response]``.
        action_mask: ``(B, A)`` mask over the response (action) tokens,
            aligned with the *last* ``A`` columns of ``sequences`` -- see
            ``Actor.forward``, where
            ``action_log_probs = log_probs[:, -action_mask.shape[1]:]``.

    Returns:
        ``(B, A)`` float tensor in ``{0, 1}``. Entry ``[i, j]`` is ``1`` when
        response token ``j`` of sample ``i`` also appears in that sample's
        prompt (i.e. it carries mutual information with the profile), else
        ``0``. With no prompt region (``A >= T``) the result is all zeros, so
        the downstream loss weight becomes a no-op (all ones).
    """
    if sequences.dim() != 2 or action_mask.dim() != 2:
        raise ValueError(f"sequences and action_mask must be 2D, got {sequences.dim()}D and {action_mask.dim()}D")

    batch, seq_len = sequences.shape
    action_len = action_mask.shape[1]

    # No prompt region -> nothing role-specific to upweight; weight is 1.0 downstream.
    if action_len <= 0 or action_len >= seq_len or sequences.numel() == 0:
        return torch.zeros(batch, max(action_len, 0), device=sequences.device, dtype=torch.float32)

    # Response tokens occupy the last A columns; everything before is the prompt/profile.
    response_ids = sequences[:, -action_len:]  # (B, A)
    prompt_ids = sequences[:, :-action_len].reshape(batch, -1)  # (B, P)

    # Per-sample profile membership via scatter: member[i, v] = (token v appears in prompt_i).
    vocab_size = int(sequences.max().item()) + 1
    member = torch.zeros(batch, vocab_size, device=sequences.device, dtype=torch.bool)
    member.scatter_(1, prompt_ids, True)
    specificity = member.gather(1, response_ids).to(torch.float32)  # (B, A)
    return specificity


def role_aware_loss_weight(
    specificity: torch.Tensor,
    advantages: torch.Tensor,
    amplify: float = 1.0,
    attenuate: float = 1.0,
) -> torch.Tensor:
    """Asymmetric, advantage-sign-aware per-token loss weight (RAPO).

    Implements the paper's gradient asymmetry: role-specific tokens are
    *amplified* under a positive advantage and *attenuated* under a negative
    one, while generic tokens (specificity 0) keep the standard unit weight
    and thus the standard PPO gradient::

        w = 1 + amplify * s             when advantage > 0   (in [1, 1 + amplify])
        w = 1 / (1 + attenuate * s)     when advantage <= 0  (in [1/(1 + attenuate), 1])

    where ``s`` is the role-specificity in ``[0, 1]``. The result is detached:
    it is a gradient coefficient, like OpenRLHF's importance-sampling
    correction, not a term that itself receives gradients.

    Args:
        specificity: ``(B, A)`` role-specificity in ``[0, 1]``.
        advantages: ``(B, A)`` per-token advantages.
        amplify: extra weight on a fully role-specific token under a positive
            advantage.
        attenuate: shrink applied to a fully role-specific token under a
            negative advantage.

    Returns:
        ``(B, A)`` positive float weight, detached from the autograd graph.
    """
    if amplify < 0 or attenuate < 0:
        raise ValueError(f"amplify and attenuate must be non-negative, got {amplify} and {attenuate}")

    s = specificity.float()
    adv = advantages.float().detach()
    pos = adv > 0
    w_pos = 1.0 + amplify * s
    w_neg = 1.0 / (1.0 + attenuate * s)
    weight = torch.where(pos, w_pos, w_neg)
    return weight.detach()


def compute_role_aware_weight(
    sequences: Optional[torch.Tensor],
    action_mask: Optional[torch.Tensor],
    advantages: torch.Tensor,
    amplify: float = 1.0,
    attenuate: float = 1.0,
) -> torch.Tensor:
    """Convenience entry point: sequences+mask -> detached per-token weight.

    Returns a ``(B, A)`` weight of ones when ``sequences`` or ``action_mask``
    is ``None`` (so callers can call unconditionally and stay no-op when the
    feature is off or inputs are unavailable).
    """
    if sequences is None or action_mask is None:
        return torch.ones_like(advantages, dtype=torch.float32)
    specificity = profile_overlap_specificity(sequences, action_mask)
    return role_aware_loss_weight(specificity, advantages, amplify=amplify, attenuate=attenuate)
