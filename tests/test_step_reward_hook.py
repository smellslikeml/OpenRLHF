"""Tests for the MRPO step-level reward hook (arxiv:2606.31825).

The unit tests exercise the pure hook directly. The integration test loads the
non-new ``Experience`` dataclass (from ``openrlhf.trainer.ppo_utils.experience``)
and drives ``apply_step_penalties`` end-to-end on a real experience batch,
mirroring how ``RemoteExperienceMaker.compute_advantages_and_returns`` calls it.

Heavy package ``__init__`` files (which pull optional deps) are bypassed by
registering stub packages with ``__path__`` pointing at the real source dirs,
the same trick used by ``tests/test_loss_aggregation.py``.
"""

import sys
import types
from pathlib import Path

import pytest
import torch

_ROOT = Path(__file__).resolve().parents[1]


def _make_pkg(name: str, path: Path) -> None:
    """Register a stub package that resolves submodules from ``path``."""
    if name not in sys.modules:
        pkg = types.ModuleType(name)
        pkg.__path__ = [str(path)]
        sys.modules[name] = pkg


# Stub the parent packages so submodule imports hit the real source files
# without executing their __init__.py (which would import optional deps).
_make_pkg("openrlhf", _ROOT / "openrlhf")
_make_pkg("openrlhf.utils", _ROOT / "openrlhf" / "utils")
_make_pkg("openrlhf.trainer", _ROOT / "openrlhf" / "trainer")
_make_pkg("openrlhf.trainer.ppo_utils", _ROOT / "openrlhf" / "trainer" / "ppo_utils")

from openrlhf.trainer.ppo_utils.experience import Experience  # noqa: E402
from openrlhf.trainer.ppo_utils.step_reward_hook import (  # noqa: E402
    StepLevelRewardPenaltyHook,
    apply_step_penalties,
    detect_step_boundaries,
)

NL = 99  # stand-in newline token id used by the fake tokenizer below


class _FakeTokenizer:
    """Minimal tokenizer: only "\n" is a single-token surface form."""

    unk_token_id = None

    def encode(self, text, add_special_tokens=True):
        return [NL] if text == "\n" else []

    def convert_tokens_to_ids(self, token):
        return NL if token == "\n" else self.unk_token_id


class TestStepLevelRewardPenaltyHook:
    """Pure hook + step-boundary detection unit tests."""

    def test_hook_no_op_on_positive_outcome(self):
        hook = StepLevelRewardPenaltyHook(decay=0.7)
        rewards = torch.full((6,), -1.0)
        out = hook.apply(rewards, outcome_reward=1.0, step_boundaries=[(0, 2), (2, 4), (4, 6)])
        assert torch.equal(out, rewards)

    def test_hook_no_op_when_boundaries_absent(self):
        hook = StepLevelRewardPenaltyHook(decay=0.7)
        rewards = torch.full((6,), -1.0)
        out = hook.apply(rewards, outcome_reward=-1.0, step_boundaries=None)
        assert torch.equal(out, rewards)

    def test_hook_penalizes_earlier_steps_most(self):
        hook = StepLevelRewardPenaltyHook(decay=0.7)
        rewards = torch.full((6,), -1.0)
        out = hook.apply(rewards, outcome_reward=-1.0, step_boundaries=[(0, 2), (2, 4), (4, 6)])

        # Latest step keeps full weight; each earlier step is attenuated by another
        # factor of decay, so the earliest step is damped the most.
        assert torch.allclose(out[0:2], torch.full((2,), -0.49))  # decay ** 2
        assert torch.allclose(out[2:4], torch.full((2,), -0.70))  # decay ** 1
        assert torch.allclose(out[4:6], torch.full((2,), -1.00))  # decay ** 0

    def test_hook_does_not_mutate_input(self):
        hook = StepLevelRewardPenaltyHook(decay=0.7)
        rewards = torch.full((6,), -1.0)
        original = rewards.clone()
        hook.apply(rewards, outcome_reward=-1.0, step_boundaries=[(0, 2), (2, 4), (4, 6)])
        assert torch.equal(rewards, original)

    def test_detect_step_boundaries_splits_on_newlines(self):
        ids = torch.tensor([1, 2, NL, 3, 4, NL, 5, 6])
        boundaries = detect_step_boundaries(ids, {NL})
        assert boundaries == [(0, 3), (3, 6), (6, 8)]

    def test_detect_step_boundaries_returns_none_without_newlines(self):
        assert detect_step_boundaries(torch.tensor([1, 2, 3, 4]), {NL}) is None

    def test_hook_uses_configured_success_threshold(self):
        # With a threshold of 1.0, a positive-but-below-threshold outcome (0.5)
        # is still treated as a failure and gets penalized.
        hook = StepLevelRewardPenaltyHook(decay=0.7, success_threshold=1.0)
        rewards = torch.full((6,), -1.0)
        out = hook.apply(rewards, outcome_reward=0.5, step_boundaries=[(0, 2), (2, 4), (4, 6)])
        assert torch.allclose(out[0:2], torch.full((2,), -0.49))  # decay ** 2
        assert torch.allclose(out[2:4], torch.full((2,), -0.70))  # decay ** 1
        assert torch.allclose(out[4:6], torch.full((2,), -1.00))  # decay ** 0

    def test_hook_leaves_at_threshold_outcome_unchanged(self):
        # outcome == threshold is treated as success (>= is inclusive).
        hook = StepLevelRewardPenaltyHook(decay=0.7, success_threshold=1.0)
        rewards = torch.full((6,), -1.0)
        out = hook.apply(rewards, outcome_reward=1.0, step_boundaries=[(0, 2), (2, 4), (4, 6)])
        assert torch.equal(out, rewards)


class TestApplyStepPenalties:
    """Integration: drives apply_step_penalties on a real Experience batch."""

    @staticmethod
    def _make_experience(rewards_row):
        """One-sample experience: 2-token prompt + 8-token response with two newlines.

        Response tokens: [1, 2, NL, 3, 4, NL, 5, 6] -> three reasoning steps.
        """
        return Experience(
            sequences=torch.tensor([[10, 11, 1, 2, NL, 3, 4, NL, 5, 6]]),
            rewards=rewards_row,
            response_length=torch.tensor([8]),
            total_length=torch.tensor([10]),
            info={},
        )

    def test_apply_step_penalties_shapes_negative_outcome_sample(self):
        exp = self._make_experience(torch.full((1, 8), -1.0))

        shaped = apply_step_penalties([exp], _Args(), tokenizer=_FakeTokenizer())

        assert shaped == 1
        rewards = exp.rewards[0]
        assert torch.allclose(rewards[0:3], torch.full((3,), -0.49))
        assert torch.allclose(rewards[3:6], torch.full((3,), -0.70))
        assert torch.allclose(rewards[6:8], torch.full((2,), -1.00))

    def test_apply_step_penalties_leaves_positive_outcome_unchanged(self):
        exp = self._make_experience(torch.full((1, 8), 1.0))

        shaped = apply_step_penalties([exp], _Args(), tokenizer=_FakeTokenizer())

        assert shaped == 0
        assert torch.allclose(exp.rewards[0], torch.full((8,), 1.0))

    def test_apply_step_penalties_is_noop_when_disabled(self):
        exp = self._make_experience(torch.full((1, 8), -1.0))
        before = exp.rewards.clone()

        class _DisabledArgs:
            class reward:
                mrpo_step_decay = None

        assert apply_step_penalties([exp], _DisabledArgs(), tokenizer=_FakeTokenizer()) == 0
        assert torch.equal(exp.rewards, before)

    def test_apply_step_penalties_honors_custom_success_threshold(self):
        # A positive-but-below-threshold outcome should still be penalized.
        exp = self._make_experience(torch.tensor([[0.0625] * 8]))  # sum = 0.5

        class _ThresholdArgs:
            class reward:
                mrpo_step_decay = 0.7
                mrpo_success_threshold = 1.0

        shaped = apply_step_penalties([exp], _ThresholdArgs(), tokenizer=_FakeTokenizer())

        assert shaped == 1
        # Each token starts at 0.0625; the three reasoning steps get decay ** 2,
        # decay ** 1, decay ** 0 respectively.
        rewards = exp.rewards[0]
        assert torch.allclose(rewards[0:3], torch.full((3,), 0.0625 * 0.49))
        assert torch.allclose(rewards[3:6], torch.full((3,), 0.0625 * 0.70))
        assert torch.allclose(rewards[6:8], torch.full((2,), 0.0625 * 1.00))

    def test_apply_step_penalties_skips_response_without_newlines(self):
        # Single-step response (no newline) cannot be decayed across steps.
        exp = Experience(
            sequences=torch.tensor([[10, 11, 1, 2, 3, 4]]),
            rewards=torch.full((1, 4), -1.0),
            response_length=torch.tensor([4]),
            total_length=torch.tensor([6]),
            info={},
        )

        assert apply_step_penalties([exp], _Args(), tokenizer=_FakeTokenizer()) == 0
        assert torch.allclose(exp.rewards[0], torch.full((4,), -1.0))

    def test_call_site_wires_in_apply_step_penalties(self):
        # Guards the wiring edit: the reward-shaping call site must invoke the hook.
        source = (_ROOT / "openrlhf" / "trainer" / "ppo_utils" / "experience_maker.py").read_text()
        assert "from openrlhf.trainer.ppo_utils.step_reward_hook import apply_step_penalties" in source
        assert "apply_step_penalties(experiences, args, self.tokenizer)" in source


class _Args:
    class reward:
        mrpo_step_decay = 0.7


if __name__ == "__main__":
    pytest.main([__file__, "-vv"])
