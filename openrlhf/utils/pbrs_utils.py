"""Optional reward-labeler hook for PPO reward collection.

Adapted from *Automating Potential-based Reward Shaping with Vision Language
Model Guidance* (arxiv:2606.27180), which learns a reward/potential signal
from lightweight VLM preferences and blends it into the reward so exploration
is shaped without inducing reward hacking.

This module ships the *labeler protocol and blending logic* that sits at
OpenRLHF's reward-collection call site
(``RemoteExperienceMaker.make_experience_batch``). It is deliberately
labeler-agnostic and free of any deep-learning dependency: a subclass
implements :meth:`PBRSLabeler.score` to turn decoded sequences into a
per-sequence signal, and :meth:`PBRSLabeler.apply` blends it with the reward
model's reward.

A concrete VLM-backed labeler -- loading a small VLM checkpoint on a device,
querying it for image/sequence preferences, and fitting a potential model --
is intentionally out of scope for the insertion-point PR that lands this
module. :class:`IdentityLabeler` is the shipped reference subclass: it makes
the labeler signal equal to the reward-model reward so the blend is a true
no-op, and it lets the protocol be exercised end-to-end without an external
checkpoint.
"""


class PBRSLabeler:
    """Blend an external labeler signal into the reward-model reward.

    ``blend`` controls how much of the reward-model reward is replaced by the
    labeler's signal:

    * ``blend = 0.0`` -- no-op: the reward-model reward is returned unchanged
      (optimal-policy-preserving, mirroring the PBRS guarantee).
    * ``blend = 1.0`` -- the labeler signal fully replaces the reward-model
      reward.
    * ``0.0 < blend < 1.0`` -- a convex combination of the two.

    The blending arithmetic uses plain Python operators, so it works on
    whatever numeric objects the caller passes (e.g. ``torch.Tensor`` at
    runtime, plain floats in tests) without importing a tensor library here.
    """

    def __init__(self, blend: float = 1.0) -> None:
        assert 0.0 <= blend <= 1.0, f"blend must be in [0.0, 1.0], got {blend}"
        self.blend = blend

    def apply(self, rewards_list, sequences_list):
        """Return blended rewards, one entry per sequence (same shapes)."""
        if self.blend == 0.0:
            return rewards_list  # cheap no-op path; preserves RM rewards exactly
        labeler_scores = self.score(rewards_list, sequences_list)
        return [
            (1.0 - self.blend) * reward + self.blend * score for reward, score in zip(rewards_list, labeler_scores)
        ]

    def score(self, rewards_list, sequences_list):
        """Subclass hook: return one score per sequence, matching reward shape.

        ``rewards_list`` is passed so a subclass can match shapes or act as a
        baseline (as :class:`IdentityLabeler` does); a learned potential
        labeler -- e.g. the VLM-derived labeler of arxiv:2606.27180 -- ignores
        it and scores from ``sequences_list`` alone.
        """
        raise NotImplementedError("PBRSLabeler subclasses must implement score()")


class IdentityLabeler(PBRSLabeler):
    """Reference no-op labeler whose signal equals the reward-model reward.

    At any ``blend`` the blend formula collapses to the input reward, so this
    is a true no-op. It serves as the blend baseline and lets the labeler
    protocol be exercised end-to-end without an external VLM checkpoint.
    """

    def score(self, rewards_list, sequences_list):
        return list(rewards_list)
