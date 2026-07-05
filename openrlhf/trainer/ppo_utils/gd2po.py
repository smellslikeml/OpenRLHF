"""Group-Dynamic reward-Decoupled Policy Optimization (GD²PO) advantage shaping.

Adapted from "GD²PO: Mitigating Multi-Reward Conflicts via Group-Dynamic
reward-Decoupled Policy Optimization", https://arxiv.org/abs/2606.16771

GD²PO extends GRPO/GDPO to the multi-reward setting. When a single rollout
earns a positive advantage on some reward dimensions and a negative advantage
on others, summing the per-dimension advantages lets the opposing signals
cancel, collapsing the effective advantage toward zero (a wasted sample).
GD²PO mitigates this with two mechanisms:

  1. Conflict-aware filtering — mask rollouts whose per-dimension advantages
     disagree in sign, so canceling signals are dropped instead of averaged.
     This is the multi-reward analogue of DAPO's dynamic sampling, which
     filters out near-zero-advantage rollouts.
  2. Query-level reweighting — scale each prompt group's update by its reward
     consensus (the fraction of the group's rollouts whose dimensions agree),
     so prompts with aligned objectives contribute more strongly.

The estimator slots into ``compute_advantages_and_returns`` in
``experience_maker.py`` next to ``rloo`` / ``group_norm`` / ``dr_grpo``. It
reads per-dimension reward scores that a multi-reward reward function writes
into ``experience.info["score"]`` (one vector of per-dimension scores per
rollout) and returns scalar shaped rewards that the rest of the GRPO advantage
path consumes unchanged. With a single reward dimension the conflict filter is
inactive and the result reduces to GRPO's group-mean baseline (``dr_grpo``).
"""

from typing import List, Tuple

import torch


def gather_reward_dims(
    experiences: List,
    indices: torch.Tensor,
    n_samples_per_prompt: int,
) -> torch.Tensor:
    """Assemble per-dimension reward scores from a batch of experiences.

    Each experience must carry ``info["score"]`` as a tensor (or list of
    tensors) covering its rollouts, with the last axis indexing reward
    dimensions. Scores are concatenated in experience order, restored to the
    original prompt order via the same ``indices`` scatter used for scalar
    rewards in ``compute_advantages_and_returns``, and reshaped to
    ``(n_prompts, n_samples_per_prompt, n_dims)``.
    """
    parts = []
    for experience in experiences:
        score = experience.info.get("score")
        if score is None:
            raise ValueError(
                "advantage_estimator='gd2po' requires per-dimension reward scores in "
                "experience.info['score']; provide a reward function that returns a "
                "vector of per-dimension scores per rollout."
            )
        if isinstance(score, (list, tuple)):
            score = torch.stack([torch.as_tensor(s).reshape(-1) for s in score])
        score = torch.as_tensor(score).reshape(len(experience.index), -1)
        parts.append(score.float())
    reward_dims = torch.cat(parts, dim=0)  # (total_rollouts, n_dims), experience order

    # Scatter back to original prompt order, mirroring `rewards[indices] = raw_rewards`.
    sorted_dims = torch.empty_like(reward_dims)
    sorted_dims[indices] = reward_dims
    return sorted_dims.reshape(-1, n_samples_per_prompt, reward_dims.size(-1))


def group_dynamic_advantages(
    reward_dims: torch.Tensor,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, dict]:
    """Compute GD²PO conflict-filtered, query-reweighted shaped rewards.

    Args:
        reward_dims: ``(n_prompts, n_samples, n_dims)`` per-dimension rewards.
        eps: magnitudes within ``[-eps, eps]`` are treated as zero when
            detecting sign disagreement, ignoring numerical dust.

    Returns:
        ``(shaped_rewards, info)`` where ``shaped_rewards`` has shape
        ``(n_prompts, n_samples)`` and ``info`` carries the diagnostics
        ``num_conflicting`` (count of masked rollouts), ``query_weight``
        (per-prompt consensus in ``[0, 1]``) and ``conflict_mask`` (bool mask
        per rollout).
    """
    # Per-dimension group advantage: subtract each dimension's within-group mean
    # (the GRPO / "group reward decoupling" baseline), computed independently per
    # reward dimension so dimensions never contaminate one another.
    adv = reward_dims - reward_dims.mean(dim=1, keepdim=True)  # (P, N, D)

    # Conflict-aware filtering: a rollout whose per-dimension advantages mix
    # positive and negative signs would cancel under summation. Mask those
    # rollouts so they contribute no (canceling) signal.
    conflict = adv.gt(eps).any(dim=-1) & adv.lt(-eps).any(dim=-1)  # (P, N)
    adv = adv.masked_fill(conflict.unsqueeze(-1), 0.0)

    shaped = adv.sum(dim=-1)  # (P, N)

    # Query-level reweighting: scale each prompt group by its reward consensus,
    # i.e. the fraction of its rollouts whose dimensions agree in sign. Aligned
    # groups keep full strength; heavily conflicted groups are dampened.
    query_weight = (~conflict).float().mean(dim=1)  # (P,)
    shaped = shaped * query_weight.unsqueeze(1)

    info = {
        "num_conflicting": int(conflict.sum().item()),
        "query_weight": query_weight,
        "conflict_mask": conflict,
    }
    return shaped, info
