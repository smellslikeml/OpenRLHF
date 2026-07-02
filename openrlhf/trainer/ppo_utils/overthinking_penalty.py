"""
Overthinking Penalty Module for RLHF Training

Adapted from "Dynamic Rollout Editing for Reducing Overthinking in RL-Trained
Reasoning Models" (https://arxiv.org/abs/2606.17890).

DRE observes that in GRPO-style RL, *successful* trajectories that keep reasoning
after a correct answer has already emerged ("overthinking") receive the same
positive sequence-level credit as the solution-reaching prefix. Because GRPO
cannot separate the verified prefix from the unnecessary continuation, this early
imbalance compounds into more severe overthinking over training.

The full DRE method edits the rollout (truncates the post-answer thinking,
regenerates a clean ending, and prefers the edited trajectory inside the same RL
group) — that requires re-invoking the generator and lives in the rollout loop.
This module delivers the credit-assignment *result* at the reward-shaping stage:
for successful trajectories only, it measures the unnecessary continuation that
trails the first emergence of the answer and softly reduces the reward in
proportion to that tail. The verified prefix is left untouched and unsuccessful
trajectories are never penalized, mirroring DRE's asymmetric intervention while
fitting the existing scalar-reward, in-place ``List[Experience]`` contract.
"""

from typing import List, Optional, Sequence, Tuple

from openrlhf.utils.logging_utils import init_logger

logger = init_logger(__name__)

# Markers that signal an answer has emerged in a reasoning trace. ``\boxed{`` is
# the canonical math-verifier answer wrapper (see openrlhf.utils.math_utils);
# ``</think>`` closes the reasoning block in thinking-style chat templates.
DEFAULT_ANSWER_MARKERS: Tuple[str, ...] = (r"\boxed{", "</think>")


def _find_boxed_end(text: str, start: int) -> int:
    """Return the index just past the closing brace of a ``\\boxed{...}`` span.

    ``start`` is the index of the opening brace of the ``\\boxed{`` marker. Falls
    back to the end of the string if the braces are unbalanced (truncated rollout).
    """
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return i + 1
    return len(text)


def find_answer_emergence(text: str, markers: Sequence[str] = DEFAULT_ANSWER_MARKERS) -> int:
    """Index of the first character *after* the answer first emerges in ``text``.

    Returns ``-1`` when no marker is present (no detectable answer, so there is no
    overthinking tail to attribute). For ``\\boxed{`` the emergence point is the
    matching closing brace so the answer itself is part of the verified prefix,
    not the penalized tail.
    """
    best = -1
    for marker in markers:
        idx = text.find(marker)
        if idx == -1:
            continue
        if marker == r"\boxed{":
            end = _find_boxed_end(text, idx + len(marker) - 1)
        else:
            end = idx + len(marker)
        if best == -1 or end < best:
            best = end
    return best


def measure_overthinking_tail(
    text: str,
    tokenizer,
    markers: Sequence[str] = DEFAULT_ANSWER_MARKERS,
) -> int:
    """Number of tokens generated after the answer first emerges in ``text``.

    Returns ``0`` when no answer marker is found or nothing trails it.
    """
    emergence = find_answer_emergence(text, markers)
    if emergence == -1:
        return 0
    tail_text = text[emergence:].strip()
    if not tail_text:
        return 0
    return len(tokenizer.encode(tail_text, add_special_tokens=False))


def compute_overthinking_penalty(tail_tokens: int, response_tokens: int, penalty_factor: float) -> float:
    """Soft, length-proportional penalty for the unnecessary continuation.

    The penalty scales with the fraction of the response spent overthinking,
    capped at ``penalty_factor`` so a single runaway tail cannot dominate the
    group-relative advantage.
    """
    if response_tokens <= 0 or tail_tokens <= 0:
        return 0.0
    ratio = min(tail_tokens / response_tokens, 1.0)
    return -ratio * penalty_factor


def _is_successful(experience, j: int) -> bool:
    """DRE intervenes only on *successful* trajectories — the source of the
    overthinking feedback loop. Prefer the verifier ``scores`` signal and fall
    back to a positive reward when scores are unavailable.
    """
    scores = getattr(experience, "scores", None)
    if scores is not None:
        return scores[j].item() > 0
    return experience.rewards[j].item() > 0


def _decode_response(experience, j: int, tokenizer) -> str:
    """Decode the response (action) tokens of sample ``j`` to text.

    Uses ``action_mask`` to select response tokens, which is robust to prompt
    length and right padding in batched experiences.
    """
    sequences = experience.sequences[j]
    action_mask = experience.action_mask[j].bool()
    # action_mask is aligned to sequences[1:] (the first token has no action).
    response_ids = sequences[1:][action_mask]
    return tokenizer.decode(response_ids.tolist(), skip_special_tokens=False)


def apply_overthinking_penalty(
    experiences: List,
    tokenizer,
    penalty_factor: float = 1.0,
    answer_markers: Optional[Sequence[str]] = None,
    min_tail_tokens: int = 0,
) -> int:
    """DRE-style overthinking penalty for successful trajectories.

    For each successful trajectory whose answer emerges before the end of the
    response, reduce the reward in proportion to the unnecessary continuation
    that trails the answer. The verified prefix is never penalized, and
    unsuccessful trajectories are left untouched.

    Args:
        experiences: List of Experience objects with ``rewards``, ``sequences``,
            ``action_mask`` and (optionally) ``scores``.
        tokenizer: Tokenizer used to decode responses and size the tail.
        penalty_factor: Maximum reward reduction for a fully-overthought response.
        answer_markers: Substrings that signal answer emergence (default
            :data:`DEFAULT_ANSWER_MARKERS`).
        min_tail_tokens: Minimum tail length before a penalty is applied, so
            normal answer formatting is not penalized.

    Returns:
        Number of trajectories that received a penalty.
    """
    if tokenizer is None:
        logger.warning("[DRE Overthinking Penalty] no tokenizer provided; skipping.")
        return 0

    markers = tuple(answer_markers) if answer_markers else DEFAULT_ANSWER_MARKERS
    total_penalized = 0

    for experience in experiences:
        response_lengths = experience.response_length
        batch_size = len(response_lengths)

        for j in range(batch_size):
            if not _is_successful(experience, j):
                continue

            response_tokens = int(response_lengths[j].item())
            if response_tokens <= 0:
                continue

            text = _decode_response(experience, j, tokenizer)
            tail_tokens = measure_overthinking_tail(text, tokenizer, markers)
            if tail_tokens < max(min_tail_tokens, 1):
                continue

            penalty = compute_overthinking_penalty(tail_tokens, response_tokens, penalty_factor)
            if penalty < 0:
                experience.rewards[j] += penalty
                total_penalized += 1

    return total_penalized
