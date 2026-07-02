"""Tests for the GD²PO advantage estimator (conflict-aware filtering + query reweighting).

These exercise the data contract used by the ``gd2po`` branch added to
``RemoteExperienceMaker.compute_advantages_and_returns`` in
``openrlhf/trainer/ppo_utils/experience_maker.py``: per-dimension reward scores
read from ``Experience.info["score"]`` are gathered into a
``(n_prompts, n_samples, n_dims)`` tensor and shaped by
``group_dynamic_advantages``. ``Experience`` is imported from the existing
(non-new) ``experience`` module to prove the integration against the real
dataclass the pipeline populates.
"""

import pytest
import torch

from openrlhf.trainer.ppo_utils.experience import Experience
from openrlhf.trainer.ppo_utils.gd2po import gather_reward_dims, group_dynamic_advantages


def _rollout(score_vec, index):
    """A single-rollout Experience carrying a vector of per-dimension scores."""
    return Experience(
        info={"score": torch.tensor([score_vec], dtype=torch.float32)},
        index=[index],
    )


def test_group_dynamic_advantages_masks_conflicting_rollout_and_reweights():
    # One prompt, four samples, two reward dimensions.
    # s0=[1,1], s1=[0,0], s2=[1,-1] (mixed sign across dims), s3=[0,0]
    reward_dims = torch.tensor([[[1.0, 1.0], [0.0, 0.0], [1.0, -1.0], [0.0, 0.0]]])  # (1, 4, 2)

    # Per-dim group means: dim0=0.5, dim1=0.0
    # Advantages: s0=[+0.5,+1.0], s1=[-0.5,0.0], s2=[+0.5,-1.0](conflict), s3=[-0.5,0.0]
    shaped, info = group_dynamic_advantages(reward_dims)

    # Only s2 mixes signs -> it is the single conflicting rollout.
    assert info["num_conflicting"] == 1
    assert torch.equal(info["conflict_mask"], torch.tensor([[False, False, True, False]]))
    # Query consensus: 3 of 4 rollouts agree -> weight 0.75.
    assert torch.allclose(info["query_weight"], torch.tensor([0.75]))
    # Pre-reweight summed advantages: [1.5, -0.5, 0.0(masked), -0.5]; scaled by 0.75.
    expected = torch.tensor([[1.125, -0.375, 0.0, -0.375]])
    assert torch.allclose(shaped, expected)


def test_single_reward_dimension_reduces_to_grpo_baseline():
    # With one dimension no sign conflict is possible, so nothing is masked and
    # the result is the plain GRPO group-mean baseline (== dr_grpo).
    reward_dims = torch.tensor([[[1.0], [0.0], [0.0], [0.0]]])  # mean 0.25
    shaped, info = group_dynamic_advantages(reward_dims)

    assert info["num_conflicting"] == 0
    assert torch.allclose(info["query_weight"], torch.tensor([1.0]))
    expected = torch.tensor([[0.75, -0.25, -0.25, -0.25]])
    assert torch.allclose(shaped, expected)


def test_gather_reward_dims_sorts_by_index_like_scalar_rewards():
    # Four single-rollout experiences, but presented out of original prompt order.
    rollouts = [
        _rollout([10.0, 20.0], index=2),  # concat pos 0 -> original pos 2
        _rollout([1.0, 2.0], index=0),  # concat pos 1 -> original pos 0
        _rollout([30.0, 40.0], index=3),  # concat pos 2 -> original pos 3
        _rollout([3.0, 4.0], index=1),  # concat pos 3 -> original pos 1
    ]
    indices = torch.tensor([2, 0, 3, 1])  # mirrors experience.index flattened

    reward_dims = gather_reward_dims(rollouts, indices, n_samples_per_prompt=4)

    assert reward_dims.shape == (1, 4, 2)
    # Scatter `sorted[indices] = concat` reorders to original positions
    # [pos1, pos3, pos0, pos2] = [[1,2],[3,4],[10,20],[30,40]].
    expected = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [10.0, 20.0], [30.0, 40.0]]])
    assert torch.allclose(reward_dims, expected)


def test_gather_reward_dims_end_to_end_through_experiences():
    # Drive the full gather -> shape path the experience_maker branch runs,
    # using real Experience objects with per-dimension scores in info["score"].
    rollouts = [_rollout([1.0, 1.0], 0), _rollout([0.0, 0.0], 1), _rollout([1.0, -1.0], 2), _rollout([0.0, 0.0], 3)]
    indices = torch.tensor([0, 1, 2, 3])

    reward_dims = gather_reward_dims(rollouts, indices, n_samples_per_prompt=4)
    shaped, info = group_dynamic_advantages(reward_dims)

    assert shaped.shape == (1, 4)
    assert info["num_conflicting"] == 1
    assert torch.allclose(shaped[0, 2], torch.tensor(0.0))  # the conflicting rollout is masked


def test_gather_reward_dims_requires_score_info():
    exp = Experience(index=[0])  # no info["score"]
    with pytest.raises(ValueError, match="per-dimension reward scores"):
        gather_reward_dims([exp], torch.tensor([0]), n_samples_per_prompt=1)
