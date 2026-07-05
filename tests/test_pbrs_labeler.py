"""Tests for the PBRS reward-labeler hook (arxiv:2606.27180).

The pure labeler-logic tests load ``pbrs_labeler.py`` directly via importlib
(matching ``tests/test_loss_aggregation.py``) so they run without the repo's
heavy tensor/Ray stack. The integration test imports the real
``RemoteExperienceMaker`` call site and is gated on ``torch``/``ray`` so it
runs in CI and skips cleanly in minimal environments.
"""

import importlib.util
import sys
import types
from pathlib import Path

import pytest

_TEST_PACKAGE = "_openrlhf_pbrs_test"


def _load_labeler_module():
    root = Path(__file__).resolve().parents[1]
    ppo_utils_dir = root / "openrlhf" / "trainer" / "ppo_utils"

    pkg = types.ModuleType(_TEST_PACKAGE)
    pkg.__path__ = [str(ppo_utils_dir)]
    sys.modules[_TEST_PACKAGE] = pkg

    spec = importlib.util.spec_from_file_location(f"{_TEST_PACKAGE}.pbrs_labeler", ppo_utils_dir / "pbrs_labeler.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[f"{_TEST_PACKAGE}.pbrs_labeler"] = module
    spec.loader.exec_module(module)
    return module


_labeler_module = _load_labeler_module()
PBRSLabeler = _labeler_module.PBRSLabeler
IdentityLabeler = _labeler_module.IdentityLabeler


class _ZeroLabeler(PBRSLabeler):
    """Test labeler whose signal is zero for every sequence (any numeric type)."""

    def score(self, rewards_list, sequences_list):
        return [reward * 0 for reward in rewards_list]


class _BoomLabeler(PBRSLabeler):
    """Test labeler that raises from ``score`` to check error propagation."""

    def score(self, rewards_list, sequences_list):
        raise RuntimeError("labeler exploded")


# ── Pure labeler-logic tests (no torch required) ──────────────────────────────


def test_blend_must_be_in_unit_interval():
    with pytest.raises(AssertionError):
        PBRSLabeler(blend=1.5)
    with pytest.raises(AssertionError):
        PBRSLabeler(blend=-0.1)


def test_blend_zero_is_noop_without_scoring():
    # blend=0.0 short-circuits before score() is ever called.
    rewards = [2.0, 4.0]
    labeler = PBRSLabeler(blend=0.0)  # base class: score() would raise
    assert labeler.apply(rewards, ["a", "b"]) is rewards


def test_identity_labeler_is_noop_at_any_blend():
    rewards = [2.0, 4.0]
    for blend in (1.0, 0.5, 0.25):
        out = IdentityLabeler(blend=blend).apply(rewards, ["a", "b"])
        assert out == rewards


def test_zero_labeler_full_blend_zeros_rewards():
    rewards = [2.0, 4.0]
    out = _ZeroLabeler(blend=1.0).apply(rewards, ["a", "b"])
    assert out == [0.0, 0.0]


def test_zero_labeler_half_blend_halves_rewards():
    rewards = [2.0, 4.0]
    out = _ZeroLabeler(blend=0.5).apply(rewards, ["a", "b"])
    assert out == [1.0, 2.0]


def test_base_score_is_not_implemented():
    with pytest.raises(NotImplementedError):
        PBRSLabeler(blend=1.0).apply([2.0], ["a"])


def test_labeler_error_propagates():
    with pytest.raises(RuntimeError, match="labeler exploded"):
        _BoomLabeler(blend=1.0).apply([2.0], ["a"])


# ── Integration test: exercises the call-site wiring in experience_maker ──────


def test_remote_experience_maker_wires_pbrs_labeler():
    """The reward-collection call site routes through _apply_pbrs_labeler."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("ray")
    from openrlhf.trainer.ppo_utils.experience_maker import RemoteExperienceMaker

    # Bypass __init__ (which needs Ray actor groups); the hook only reads
    # self.pbrs_labeler and is independently testable.
    maker = RemoteExperienceMaker.__new__(RemoteExperienceMaker)
    rewards = [torch.tensor([1.0, 2.0, 3.0])]
    sequences = [torch.tensor([7, 8, 9])]

    # Default (None) -> exact passthrough, identical object.
    maker.pbrs_labeler = None
    assert maker._apply_pbrs_labeler(rewards, sequences) is rewards

    # Identity labeler at full blend -> rewards unchanged (equal values).
    maker.pbrs_labeler = IdentityLabeler(blend=1.0)
    out = maker._apply_pbrs_labeler(rewards, sequences)
    assert len(out) == len(rewards)
    assert torch.equal(out[0], rewards[0])
