import importlib.util
import sys
import types
from pathlib import Path

import torch


def _load_models_package():
    """Load the openrlhf.models submodules needed for CFPO in isolation.

    Mirrors tests/test_loss_aggregation.py: build a throwaway package whose
    __path__ points at openrlhf/models so the modules' relative imports
    (``from .utils``, ``from .loss``) resolve, without importing the heavy
    openrlhf.models package __init__ (which pulls in deepspeed / transformers).
    """
    # utils.py imports deepspeed at module load only for ZeRO3 helpers we don't
    # exercise here; stub it so the import succeeds in a CPU-only test env.
    sys.modules.setdefault("deepspeed", types.ModuleType("deepspeed"))

    root = Path(__file__).resolve().parents[1]
    models_dir = root / "openrlhf" / "models"

    pkg_name = "_openrlhf_cfpo_test"
    pkg = types.ModuleType(pkg_name)
    pkg.__path__ = [str(models_dir)]
    sys.modules[pkg_name] = pkg

    modules = {}
    for name in ("utils", "loss", "counterfactual_grounding"):
        spec = importlib.util.spec_from_file_location(f"{pkg_name}.{name}", models_dir / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[f"{pkg_name}.{name}"] = module
        spec.loader.exec_module(module)
        modules[name] = module
    return modules


_models = _load_models_package()
_utils = _models["utils"]
_loss = _models["loss"]
_cfg = _models["counterfactual_grounding"]


def test_suppress_visual_inputs_zeros_pixels_keeps_metadata():
    mm_inputs = {
        "pixel_values": torch.randn(4, 8),
        "image_grid_thw": torch.tensor([[1, 2, 2]]),
    }
    out = _cfg.suppress_visual_inputs(mm_inputs)

    # Visual content is suppressed but the shape (placeholder bookkeeping) is kept.
    assert out["pixel_values"].shape == mm_inputs["pixel_values"].shape
    assert torch.count_nonzero(out["pixel_values"]) == 0
    # Grid metadata is left untouched so the counterfactual forward still emits
    # the same number of visual placeholder embeddings.
    assert torch.equal(out["image_grid_thw"], mm_inputs["image_grid_thw"])
    # The original tensor must not be mutated in place.
    assert torch.count_nonzero(mm_inputs["pixel_values"]) > 0


def test_suppress_visual_inputs_no_visual_returns_empty():
    assert _cfg.suppress_visual_inputs({}) == {}
    # Text-only mm metadata (no pixel content) -> nothing to suppress.
    assert _cfg.suppress_visual_inputs({"image_grid_thw": torch.tensor([[1, 1, 1]])}) == {}


def test_grounding_loss_composes_existing_kl_and_aggregation():
    """The discrepancy must equal compose(compute_approx_kl, aggregate_loss)
    from the existing (non-new) openrlhf.models modules."""
    torch.manual_seed(0)
    factual = torch.randn(2, 5)
    counterfactual = torch.randn(2, 5)
    action_mask = torch.ones(2, 5)

    got = _cfg.counterfactual_grounding_loss(factual, counterfactual, action_mask, kl_estimator="k3")

    expected_kl = _utils.compute_approx_kl(factual, counterfactual, kl_estimator="k3")
    expected = _loss.aggregate_loss(expected_kl, action_mask, token_level_loss=True)

    assert torch.allclose(got, expected)


def test_grounding_loss_nonnegative_and_zero_when_identical():
    factual = torch.randn(3, 4)
    action_mask = torch.ones(3, 4)

    # Identical factual/counterfactual -> no discrepancy.
    same = _cfg.counterfactual_grounding_loss(factual, factual.clone(), action_mask)
    assert torch.allclose(same, torch.zeros_like(same), atol=1e-6)

    # k3 is a non-negative KL estimator, so the grounding discrepancy is >= 0.
    diff = _cfg.counterfactual_grounding_loss(factual, torch.randn(3, 4), action_mask)
    assert diff.item() >= 0.0


def test_trainer_wiring_maximizes_discrepancy():
    """Reproduce the call-site arithmetic in ppo_actor.training_step:
    ``loss = loss - cfpo_coef * discrepancy``. A positive coef must lower the
    objective as grounding discrepancy grows (i.e. it is maximized)."""
    factual = torch.zeros(2, 4)
    action_mask = torch.ones(2, 4)
    base_loss = torch.tensor(1.0)
    cfpo_coef = 0.1

    weak = _cfg.counterfactual_grounding_loss(factual, factual + 0.1, action_mask)
    strong = _cfg.counterfactual_grounding_loss(factual, factual + 1.0, action_mask)
    assert strong > weak

    loss_weak = base_loss - cfpo_coef * weak
    loss_strong = base_loss - cfpo_coef * strong
    assert loss_strong < loss_weak
