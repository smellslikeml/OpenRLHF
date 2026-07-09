"""Tests for role-aware policy-loss weighting (RAPO, arXiv:2606.27025).

Loads ``openrlhf/models/loss.py`` (the existing, edited call-site module)
plus its sibling ``role_aware_weighting.py`` and ``utils.py`` through a
standalone fake package, exactly like ``tests/test_loss_aggregation.py``,
so we exercise the integrated ``PolicyLoss`` without importing the heavy
``openrlhf.models`` package (which pulls in transformers/peft/deepspeed).
"""

import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch

_TEST_PACKAGE = "_openrlhf_role_aware_test"


def _load_modules():
    root = Path(__file__).resolve().parents[1]
    models_dir = root / "openrlhf" / "models"

    pkg = types.ModuleType(_TEST_PACKAGE)
    pkg.__path__ = [str(models_dir)]
    sys.modules[_TEST_PACKAGE] = pkg

    for name in ("utils", "role_aware_weighting", "loss"):
        spec = importlib.util.spec_from_file_location(f"{_TEST_PACKAGE}.{name}", models_dir / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[f"{_TEST_PACKAGE}.{name}"] = module
        spec.loader.exec_module(module)

    return sys.modules[f"{_TEST_PACKAGE}.loss"], sys.modules[f"{_TEST_PACKAGE}.role_aware_weighting"]


_loss_module, _role_module = _load_modules()
PolicyLoss = _loss_module.PolicyLoss
profile_overlap_specificity = _role_module.profile_overlap_specificity
role_aware_loss_weight = _role_module.role_aware_loss_weight


# sequences layout: prompt = [pad(0), 5, 5, 9], response = [5, 7, 5]
# Token 5 appears in the prompt (role-specific); token 7 does not (generic).
_SEQ = torch.tensor([[0, 5, 5, 9, 5, 7, 5]])
_MASK = torch.tensor([[1.0, 1.0, 1.0]])


def test_profile_overlap_specificity_marks_prompt_overlap():
    spec = profile_overlap_specificity(_SEQ, _MASK)
    # response [5, 7, 5]: 5 is in the prompt -> 1, 7 is not -> 0, 5 -> 1
    assert torch.equal(spec, torch.tensor([[1.0, 0.0, 1.0]]))


def test_profile_overlap_specificity_no_prompt_is_noop():
    # A >= T means there is no prompt region -> all zeros -> weight is 1 downstream.
    seq = torch.tensor([[5, 7, 5]])
    mask = torch.tensor([[1.0, 1.0, 1.0]])
    assert torch.equal(profile_overlap_specificity(seq, mask), torch.zeros(1, 3))


def test_role_aware_weight_amplifies_positive_attenuates_negative():
    specificity = torch.tensor([[1.0, 0.0, 1.0]])

    w_pos = role_aware_loss_weight(specificity, torch.tensor([[1.0, 1.0, 1.0]]), amplify=1.0, attenuate=1.0)
    assert torch.allclose(w_pos, torch.tensor([[2.0, 1.0, 2.0]]))

    w_neg = role_aware_loss_weight(specificity, torch.tensor([[-1.0, -1.0, -1.0]]), amplify=1.0, attenuate=1.0)
    assert torch.allclose(w_neg, torch.tensor([[0.5, 1.0, 0.5]]))

    # Generic tokens (specificity 0) are never reweighted, regardless of advantage sign.
    assert w_pos[0, 1] == 1.0
    assert w_neg[0, 1] == 1.0

    # The weight is detached: a gradient coefficient, not a learned term.
    assert not w_pos.requires_grad


def test_role_aware_weight_rejects_negative_coeffs():
    with pytest.raises(ValueError):
        role_aware_loss_weight(torch.tensor([[1.0]]), torch.tensor([[1.0]]), amplify=-1.0)


def test_policy_loss_role_aware_alters_loss_by_advantage_sign():
    # ratio = exp(0) = 1, so the unweighted per-token PPO loss is simply -advantage.
    action_len = _MASK.shape[1]
    log_probs = torch.zeros(1, action_len)
    old_log_probs = torch.zeros(1, action_len)

    base = PolicyLoss(policy_loss_type="ppo")
    ra = PolicyLoss(policy_loss_type="ppo", enable_role_aware_weighting=True, role_amplify=1.0, role_attenuate=1.0)

    # Positive advantage: role-specific tokens are amplified -> |loss| grows vs vanilla PPO.
    adv_pos = torch.tensor([[1.0, 1.0, 1.0]])
    loss_base_pos, *_ = base(log_probs, old_log_probs, adv_pos, action_mask=_MASK)
    loss_ra_pos, *_ = ra(log_probs, old_log_probs, adv_pos, action_mask=_MASK, sequences=_SEQ)
    assert loss_ra_pos.abs() > loss_base_pos.abs()

    # Negative advantage: role-specific tokens are attenuated -> |loss| shrinks vs vanilla PPO.
    adv_neg = torch.tensor([[-1.0, -1.0, -1.0]])
    loss_base_neg, *_ = base(log_probs, old_log_probs, adv_neg, action_mask=_MASK)
    loss_ra_neg, *_ = ra(log_probs, old_log_probs, adv_neg, action_mask=_MASK, sequences=_SEQ)
    assert loss_ra_neg.abs() < loss_base_neg.abs()


def test_policy_loss_default_is_backward_compatible():
    # With the feature off (default), passing sequences must not change the loss.
    action_len = _MASK.shape[1]
    log_probs = torch.zeros(1, action_len)
    old_log_probs = torch.zeros(1, action_len)
    adv = torch.tensor([[1.0, 1.0, 1.0]])

    pl = PolicyLoss(policy_loss_type="ppo")
    loss_without_seq, *_ = pl(log_probs, old_log_probs, adv, action_mask=_MASK)
    loss_with_seq, *_ = pl(log_probs, old_log_probs, adv, action_mask=_MASK, sequences=_SEQ)
    assert torch.allclose(loss_without_seq, loss_with_seq)


def test_policy_loss_role_aware_without_sequences_is_noop():
    # Enabled but no sequences -> weight falls back to ones -> identical to vanilla PPO.
    action_len = _MASK.shape[1]
    log_probs = torch.zeros(1, action_len)
    old_log_probs = torch.zeros(1, action_len)
    adv = torch.tensor([[1.0, 1.0, 1.0]])

    base = PolicyLoss(policy_loss_type="ppo")
    ra = PolicyLoss(policy_loss_type="ppo", enable_role_aware_weighting=True)
    loss_base, *_ = base(log_probs, old_log_probs, adv, action_mask=_MASK)
    loss_ra, *_ = ra(log_probs, old_log_probs, adv, action_mask=_MASK)
    assert torch.allclose(loss_base, loss_ra)
