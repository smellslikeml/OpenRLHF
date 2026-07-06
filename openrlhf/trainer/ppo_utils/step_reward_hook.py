"""Step-level reward shaping for PPO/GRPO training.

Ships the step-level penalty mechanism from the MRPO paper (arxiv:2606.31825,
"Breaking Failure Cascades") as one self-contained hook. **This is not the full
MRPO training procedure** — only the step-level penalty component, which is the
paper's discrete, portable piece that generalises beyond the medical-VLM
context to any PPO/GRPO chain-of-thought training loop.

MRPO observes that when an outcome is wrong, errors tend to cascade from
early reasoning steps. To break those cascades it shapes the per-token reward
with an exponentially decaying factor across reasoning steps: the latest step
keeps its full weight and each earlier step is attenuated by another factor of
``decay``, so the earliest step is damped the most. Successful rollouts
(``outcome_reward >= success_threshold``, default threshold ``0.0`` to match
the paper's binary-outcome framing) are returned unchanged, leaving the credit
signal on correct trajectories intact. The threshold is opt-in for use with
continuous reward models that are not zero-centered.

Step boundaries are detected from single-token newlines as a first-cut
approximation of reasoning-step delimiters. A learned step verifier is the
natural upgrade path noted in the paper — see :func:`detect_step_boundaries`
if you plan to plug in a real verifier.
"""

from typing import List, Optional, Sequence, Tuple

import torch

from openrlhf.utils.logging_utils import init_logger

logger = init_logger(__name__)


class StepLevelRewardPenaltyHook:
    """Apply MRPO's exponentially-decaying step penalty to per-token rewards.

    Args:
        decay: Per-step attenuation factor in ``(0, 1)``. The latest step is
            scaled by ``decay ** 0 == 1`` and each earlier step by an additional
            factor of ``decay``, so the earliest step is scaled by
            ``decay ** (num_steps - 1)``. The paper (§4.2) reports the strongest
            gains around ``decay == 0.7`` on GSM8K-style benchmarks; below 0.5
            the earliest steps' credit signal collapses too aggressively, and
            above 0.9 the decay is barely distinguishable from no shaping.
        success_threshold: Outcome cutoff for treating a rollout as successful.
            The paper uses binary correctness with ``0`` as the natural cutoff;
            continuous reward models emitting logits that are not zero-centered
            (e.g., a preference RM whose "correct" outputs cluster around 0.4)
            need a matching threshold. A rollout is left unchanged iff
            ``outcome_reward >= success_threshold``.
    """

    def __init__(self, decay: float = 0.7, success_threshold: float = 0.0):
        if not 0.0 < decay < 1.0:
            raise ValueError(f"decay must be in (0, 1), got {decay}")
        self.decay = decay
        self.success_threshold = success_threshold

    def apply(
        self,
        per_token_reward: torch.Tensor,
        outcome_reward: float,
        step_boundaries: Optional[Sequence[Tuple[int, int]]],
    ) -> torch.Tensor:
        """Return a (possibly) shaped copy of ``per_token_reward``.

        No-op when the rollout is successful
        (``outcome_reward >= self.success_threshold``) or when no step
        boundaries are available.
        """
        if step_boundaries is None or outcome_reward >= self.success_threshold:
            return per_token_reward

        modified = per_token_reward.clone()
        # Iterate from the latest step back to the earliest: i == 0 for the last
        # step (full weight), increasing toward the first step (most attenuation).
        for i, (start, end) in enumerate(reversed(list(step_boundaries))):
            modified[start:end] = modified[start:end] * (self.decay**i)
        return modified


def detect_step_boundaries(
    response_token_ids: torch.Tensor, newline_token_ids: Sequence[int]
) -> Optional[List[Tuple[int, int]]]:
    """Split a response into ``(start, end)`` token ranges at newline tokens.

    A newline ends the step it belongs to. Returns ``None`` when fewer than two
    steps are found, since a single step has no earlier/later contrast to decay
    across.
    """
    total = len(response_token_ids)
    if total == 0:
        return None

    boundaries: List[Tuple[int, int]] = []
    start = 0
    # Materialize once as a Python list: per-token ``.item()`` calls trigger a
    # device-to-host sync on GPU tensors, and even on CPU tensors are much
    # slower than iterating a native list. Set lookup keeps membership O(1).
    token_ids = response_token_ids.tolist()
    newline_set = set(newline_token_ids)
    for idx, token_id in enumerate(token_ids):
        if token_id in newline_set:
            boundaries.append((start, idx + 1))
            start = idx + 1
    if start < total:
        boundaries.append((start, total))

    if len(boundaries) < 2:
        return None
    return boundaries


def resolve_newline_token_ids(tokenizer) -> List[int]:
    """Collect single-token ids whose decoding yields a newline.

    Different vocabularies represent newlines differently (``"\\n"``,
    ``"\\n\\n"``, a leading space); each surface form the tokenizer encodes as a
    single token is picked up. Direct vocab entries are also probed.
    """
    newline_ids = set()
    for surface in ("\n", "\n\n", " \n"):
        encoded = tokenizer.encode(surface, add_special_tokens=False)
        if len(encoded) == 1:
            newline_ids.add(int(encoded[0]))

    unk_token_id = getattr(tokenizer, "unk_token_id", None)
    for name in ("\n", "<0x0A>"):
        token_id = tokenizer.convert_tokens_to_ids(name)
        if token_id is not None and token_id >= 0 and token_id != unk_token_id:
            newline_ids.add(int(token_id))
    return sorted(newline_ids)


def _response_token_ids(experience, j: int) -> torch.Tensor:
    """Slice the generated (response) token ids for sample ``j`` of an experience."""
    total_len = int(experience.total_length[j].item())
    response_len = int(experience.response_length[j].item())
    prompt_len = total_len - response_len
    return experience.sequences[j, prompt_len:total_len]


def apply_step_penalties(experiences, args, tokenizer=None) -> int:
    """Shape per-token rewards with MRPO step penalties (mutates in place).

    Opt-in: active only when ``args.reward.mrpo_step_decay`` is set to a value
    in ``(0, 1)``. Also reads ``args.reward.mrpo_success_threshold`` (default
    ``0.0``) to define the outcome cutoff that separates successful rollouts
    (left unchanged) from failed ones (penalized). Otherwise a no-op
    (returning ``0``), so callers that do not enable it see no behavior change.

    Should run before :func:`apply_length_penalties`, whose own ``info["reward"]``
    sync then captures the shaped rewards for logging.

    Args:
        experiences: List of batched ``Experience`` objects (mutated in place).
        args: Training args; ``args.reward.mrpo_step_decay`` and
            ``args.reward.mrpo_success_threshold`` are read.
        tokenizer: Tokenizer used to locate newline step delimiters.

    Returns:
        Number of samples whose per-token rewards were shaped.
    """
    decay = getattr(args.reward, "mrpo_step_decay", None)
    if decay is None or float(decay) <= 0.0:
        return 0
    decay = float(decay)
    if decay >= 1.0:
        logger.warning("[MRPO Step Penalty] mrpo_step_decay=%s is >= 1.0; skipping (no attenuation)", decay)
        return 0
    if tokenizer is None:
        logger.warning("[MRPO Step Penalty] enabled (decay=%s) but no tokenizer provided; skipping", decay)
        return 0

    newline_ids = resolve_newline_token_ids(tokenizer)
    if not newline_ids:
        logger.warning("[MRPO Step Penalty] tokenizer exposes no single-token newline; skipping")
        return 0

    success_threshold = float(getattr(args.reward, "mrpo_success_threshold", 0.0))
    hook = StepLevelRewardPenaltyHook(decay=decay, success_threshold=success_threshold)
    total_samples = sum(len(exp.rewards) for exp in experiences)
    shaped = 0
    for experience in experiences:
        rewards = experience.rewards
        for j in range(len(rewards)):
            per_token = rewards[j]
            if per_token.dim() == 0:
                continue  # per-episode scalar reward: nothing to shape per-token
            outcome = float(per_token.sum().item())
            if outcome >= success_threshold:
                continue  # successful rollout: leave the credit signal intact
            response_ids = _response_token_ids(experience, j)
            if len(response_ids) != len(per_token):
                continue  # token/reward misalignment: skip rather than corrupt
            boundaries = detect_step_boundaries(response_ids, newline_ids)
            if boundaries is None:
                continue
            rewards[j] = hook.apply(per_token, outcome, boundaries)
            shaped += 1

    logger.info("[MRPO Step Penalty] %d/%d samples shaped, decay=%s", shaped, total_samples, decay)
    return shaped
