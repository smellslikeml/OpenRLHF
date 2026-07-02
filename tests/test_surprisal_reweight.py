import importlib.util
import logging
import sys
import types
import unittest
from pathlib import Path

import torch

_TEST_PACKAGE = "_openrlhf_surprisal_test"


def _stub_openrlhf_utils():
    """Register lightweight stubs for the openrlhf.utils symbols that
    experience.py / surprisal_reweight.py import absolutely, so neither test
    drags in transformers/ray/deepspeed via the real openrlhf.utils package."""
    if "openrlhf.utils.utils" in sys.modules:
        return

    openrlhf_pkg = sys.modules.setdefault("openrlhf", types.ModuleType("openrlhf"))
    openrlhf_pkg.__path__ = []
    utils_pkg = types.ModuleType("openrlhf.utils")
    utils_pkg.__path__ = []
    sys.modules["openrlhf.utils"] = utils_pkg

    utils_mod = types.ModuleType("openrlhf.utils.utils")
    utils_mod.zero_pad_sequences = lambda *a, **k: None  # not exercised by these tests
    sys.modules["openrlhf.utils.utils"] = utils_mod

    logging_mod = types.ModuleType("openrlhf.utils.logging_utils")
    logging_mod.init_logger = lambda name: logging.getLogger(name)
    sys.modules["openrlhf.utils.logging_utils"] = logging_mod


def _load_ppo_utils_modules():
    """Load experience.py (existing call-site module) and surprisal_reweight.py
    in isolation, mirroring tests/test_loss_aggregation.py so the heavy package
    __init__ chain (ray, vllm, deepspeed) is not imported."""
    _stub_openrlhf_utils()
    root = Path(__file__).resolve().parents[1]
    ppo_utils_dir = root / "openrlhf" / "trainer" / "ppo_utils"

    pkg = types.ModuleType(_TEST_PACKAGE)
    pkg.__path__ = [str(ppo_utils_dir)]
    sys.modules[_TEST_PACKAGE] = pkg

    modules = {}
    for name in ("experience", "surprisal_reweight"):
        spec = importlib.util.spec_from_file_location(f"{_TEST_PACKAGE}.{name}", ppo_utils_dir / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[f"{_TEST_PACKAGE}.{name}"] = module
        spec.loader.exec_module(module)
        modules[name] = module
    return modules


_modules = _load_ppo_utils_modules()
Experience = _modules["experience"].Experience
surprisal_reweight = _modules["surprisal_reweight"]


class _AdvantageCfg:
    def __init__(self, **kwargs):
        self.surprisal_reweight = False
        self.surprisal_reweight_quantile = 0.8
        self.surprisal_reweight_factor = 1.5
        self.surprisal_reweight_sign_aware = True
        for k, v in kwargs.items():
            setattr(self, k, v)


class _Algo:
    def __init__(self, advantage):
        self.advantage = advantage


class _Args:
    def __init__(self, advantage):
        self.algo = _Algo(advantage)


def _make_experience(advantages, log_probs, mask):
    """Build a real Experience (non-new module) with the fields the reweighting
    path reads, matching what compute_advantages_and_returns produces."""
    advantages = torch.tensor(advantages, dtype=torch.float32)
    exp = Experience(
        action_log_probs=torch.tensor(log_probs, dtype=torch.float32),
        action_mask=torch.tensor(mask, dtype=torch.bool),
        advantages=advantages,
    )
    exp.rewards = torch.zeros(advantages.shape[0])
    exp.info = {"reward": exp.rewards.clone()}
    return exp


class TestSurprisalReweight(unittest.TestCase):
    def test_disabled_is_noop(self):
        args = _Args(_AdvantageCfg(surprisal_reweight=False))
        # surprisal = -log_prob; token at col 0 has the highest surprisal.
        exp = _make_experience(
            advantages=[[1.0, 1.0, 1.0, 1.0]],
            log_probs=[[-5.0, -1.0, -0.5, -0.1]],
            mask=[[1, 1, 1, 1]],
        )
        before = exp.advantages.clone()
        surprisal_reweight.apply_surprisal_reweighting([exp], args)
        self.assertTrue(torch.equal(exp.advantages, before))

    def test_high_surprisal_positive_advantage_is_boosted(self):
        args = _Args(
            _AdvantageCfg(surprisal_reweight=True, surprisal_reweight_quantile=0.75, surprisal_reweight_factor=2.0)
        )
        exp = _make_experience(
            advantages=[[1.0, 1.0, 1.0, 1.0]],
            log_probs=[[-5.0, -1.0, -0.5, -0.1]],  # token 0 is the most surprising
            mask=[[1, 1, 1, 1]],
        )
        surprisal_reweight.apply_surprisal_reweighting([exp], args)
        # Only the highest-surprisal positive-advantage token is reweighted.
        self.assertEqual(exp.advantages[0, 0].item(), 2.0)
        self.assertEqual(exp.advantages[0, 1].item(), 1.0)
        self.assertEqual(exp.advantages[0, 3].item(), 1.0)
        # Diagnostics are attached for logging.
        self.assertIn("surprisal_critical_frac", exp.info)

    def test_sign_aware_leaves_negative_advantages_untouched(self):
        args = _Args(
            _AdvantageCfg(surprisal_reweight=True, surprisal_reweight_quantile=0.5, surprisal_reweight_factor=3.0)
        )
        exp = _make_experience(
            advantages=[[-1.0, -1.0, 1.0, 1.0]],
            log_probs=[[-9.0, -8.0, -0.5, -0.1]],  # the two surprising tokens have negative advantage
            mask=[[1, 1, 1, 1]],
        )
        surprisal_reweight.apply_surprisal_reweighting([exp], args)
        # High-surprisal tokens (cols 0,1) are negative-advantage -> untouched by sign-aware mode.
        self.assertEqual(exp.advantages[0, 0].item(), -1.0)
        self.assertEqual(exp.advantages[0, 1].item(), -1.0)

    def test_masked_tokens_are_ignored_for_quantile_and_reweight(self):
        args = _Args(
            _AdvantageCfg(surprisal_reweight=True, surprisal_reweight_quantile=0.5, surprisal_reweight_factor=2.0)
        )
        exp = _make_experience(
            advantages=[[1.0, 1.0]],
            log_probs=[[-100.0, -0.1]],  # col 0 would be hugely surprising...
            mask=[[0, 1]],  # ...but it is masked out, so it must not be reweighted
        )
        surprisal_reweight.apply_surprisal_reweighting([exp], args)
        self.assertEqual(exp.advantages[0, 0].item(), 1.0)  # masked token untouched

    def test_call_site_invokes_reweighting(self):
        """The advantage path edit in experience_maker.py must call into this module.
        Verify the wiring by source-inspecting the existing call-site module without
        importing its heavy dependencies."""
        root = Path(__file__).resolve().parents[1]
        source = (root / "openrlhf" / "trainer" / "ppo_utils" / "experience_maker.py").read_text()
        self.assertIn("apply_surprisal_reweighting", source)
        self.assertIn(
            "from openrlhf.trainer.ppo_utils.surprisal_reweight import apply_surprisal_reweighting", source
        )


if __name__ == "__main__":
    unittest.main()
