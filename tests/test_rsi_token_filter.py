import importlib.util
import math
import sys
import types
from pathlib import Path

import torch

_TEST_PACKAGE = "_openrlhf_rsi_test"


def _load_modules():
    root = Path(__file__).resolve().parents[1]
    models_dir = root / "openrlhf" / "models"

    pkg = types.ModuleType(_TEST_PACKAGE)
    pkg.__path__ = [str(models_dir)]
    sys.modules[_TEST_PACKAGE] = pkg

    modules = {}
    for name in ("utils", "token_surprisal_filter", "loss"):
        spec = importlib.util.spec_from_file_location(f"{_TEST_PACKAGE}.{name}", models_dir / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[f"{_TEST_PACKAGE}.{name}"] = module
        spec.loader.exec_module(module)
        modules[name] = module
    return modules


_modules = _load_modules()
PolicyLoss = _modules["loss"].PolicyLoss
relative_surprisal_index = _modules["token_surprisal_filter"].relative_surprisal_index
rsi_select_mask = _modules["token_surprisal_filter"].rsi_select_mask


def test_relative_surprisal_index_couples_surprisal_with_entropy():
    entropy = torch.tensor([[1.0, 1.0]])
    selected_log_probs = torch.tensor([[math.log(0.5), math.log(0.01)]])  # surprisal ~= [0.693, 4.605]

    rsi = relative_surprisal_index(entropy, selected_log_probs)

    assert torch.allclose(rsi, torch.tensor([[0.6931, 4.6052]], dtype=torch.float32), atol=1e-3)
    # The index is a fixed per-step schedule, not a learned quantity.
    assert not rsi.requires_grad


def test_rsi_select_mask_keeps_typical_drops_redundant_and_tail():
    # token 0: RSI 1.0 (typical, keep); token 1: RSI 0.02 (redundant, drop);
    # token 2: RSI 5.0 (unstable tail, drop).
    entropy = torch.tensor([[1.0, 0.5, 1.0]])
    selected_log_probs = torch.tensor([[-1.0, -0.01, -5.0]])

    mask, rsi = rsi_select_mask(entropy, selected_log_probs, low=0.5, high=2.0)

    assert torch.equal(mask, torch.tensor([[True, False, False]]))
    assert torch.allclose(rsi, torch.tensor([[1.0, 0.02, 5.0]], dtype=torch.float32), atol=1e-3)


def test_policy_loss_rsi_filter_zeros_tail_token_contribution():
    # ratio == 1 (log_probs == old_log_probs) so the per-token PPO loss is -advantage before
    # filtering. Token A is typical (kept), token B is a high-surprisal tail (dropped).
    log_probs = torch.tensor([[math.log(0.5), math.log(0.01)]])
    old_log_probs = log_probs.detach().clone()
    advantages = torch.tensor([[1.0, 1.0]])
    entropy = torch.tensor([[1.0, 1.0]])  # RSI = [0.693, 4.605]
    mask = torch.tensor([[1.0, 1.0]])

    unfiltered = PolicyLoss(clip_eps_low=0.2, clip_eps_high=0.2)
    filtered = PolicyLoss(
        clip_eps_low=0.2,
        clip_eps_high=0.2,
        enable_rsi_filter=True,
        rsi_interval=[0.5, 2.0],
    )

    full_loss, *_ = unfiltered(log_probs, old_log_probs, advantages, action_mask=mask)
    filtered_loss, *_ = filtered(log_probs, old_log_probs, advantages, action_mask=mask, entropy=entropy)

    # Both tokens contribute -1 -> mean -1.0; with the tail token dropped -> mean -0.5.
    assert torch.allclose(full_loss, torch.tensor(-1.0))
    assert torch.allclose(filtered_loss, torch.tensor(-0.5))


def test_policy_loss_rsi_filter_requires_entropy():
    log_probs = torch.tensor([[math.log(0.5)]])
    old_log_probs = log_probs.detach().clone()
    advantages = torch.tensor([[1.0]])
    mask = torch.tensor([[1.0]])

    filtered = PolicyLoss(enable_rsi_filter=True, rsi_interval=[0.5, 2.0])
    try:
        filtered(log_probs, old_log_probs, advantages, action_mask=mask)
    except ValueError:
        return
    raise AssertionError("Expected ValueError when entropy is missing with rsi filter enabled")
