"""Tests for the DRE overthinking penalty and its wiring into apply_length_penalties.

The full ``openrlhf`` package pulls heavy deps (torch, sympy, transformers) through
``openrlhf.utils.__init__``. To keep these tests focused on the penalty logic we
load the two penalty modules directly from their files with a stubbed
``openrlhf.utils.logging_utils``, mirroring the importlib pattern used by
``tests/test_loss_aggregation.py``.
"""

import importlib.util
import sys
import types
from pathlib import Path

import pytest

_PPO_UTILS = Path(__file__).resolve().parents[1] / "openrlhf" / "trainer" / "ppo_utils"

_MANAGED_MODULES = (
    "openrlhf",
    "openrlhf.utils",
    "openrlhf.utils.logging_utils",
    "openrlhf.trainer",
    "openrlhf.trainer.ppo_utils",
    "openrlhf.trainer.ppo_utils.overthinking_penalty",
    "openrlhf.trainer.ppo_utils.length_penalty",
)


def _exec_module(name):
    full_name = f"openrlhf.trainer.ppo_utils.{name}"
    spec = importlib.util.spec_from_file_location(full_name, _PPO_UTILS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[full_name] = module
    spec.loader.exec_module(module)
    return module


def _load_penalty_modules():
    """Load the two penalty modules with stubbed deps, restoring sys.modules afterward.

    The real ``openrlhf.utils`` package pulls torch/sympy. We temporarily stub the
    package tree so the modules import cleanly, then restore sys.modules so other
    test modules (which import the real ``openrlhf.cli``) are unaffected.
    """
    saved = {name: sys.modules.get(name) for name in _MANAGED_MODULES}
    try:
        for name in ("openrlhf", "openrlhf.utils", "openrlhf.trainer", "openrlhf.trainer.ppo_utils"):
            pkg = types.ModuleType(name)
            pkg.__path__ = []  # mark as package
            sys.modules[name] = pkg
        logging_stub = types.ModuleType("openrlhf.utils.logging_utils")
        logging_stub.init_logger = lambda *args, **kwargs: __import__("logging").getLogger("test")
        sys.modules["openrlhf.utils.logging_utils"] = logging_stub

        overthinking = _exec_module("overthinking_penalty")
        # length_penalty imports apply_overthinking_penalty -> exercises the real wiring edit.
        length_penalty = _exec_module("length_penalty")
        return overthinking, length_penalty
    finally:
        for name, mod in saved.items():
            if mod is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = mod


overthinking, length_penalty = _load_penalty_modules()


class WhitespaceTokenizer:
    """Minimal stand-in tokenizer: one token per whitespace-separated word."""

    def encode(self, text, add_special_tokens=False):
        return text.split()

    def decode(self, ids, skip_special_tokens=False):
        return " ".join(str(i) for i in ids)


# ── Pure detection logic (runs without torch) ──


def test_find_answer_emergence_boxed_excludes_answer_from_tail():
    text = "let me think \\boxed{42} and then keep rambling"
    idx = overthinking.find_answer_emergence(text)
    # Emergence is just after the closing brace, so "42" is part of the verified prefix.
    assert text[:idx].endswith("\\boxed{42}")
    assert text[idx:].strip() == "and then keep rambling"


def test_find_answer_emergence_picks_earliest_marker():
    text = "reasoning </think> the answer is \\boxed{7}"
    idx = overthinking.find_answer_emergence(text)
    assert text[:idx].endswith("</think>")


def test_find_answer_emergence_absent_returns_minus_one():
    assert overthinking.find_answer_emergence("no answer here at all") == -1


def test_measure_overthinking_tail_counts_post_answer_tokens():
    tok = WhitespaceTokenizer()
    text = "think \\boxed{42} aa bb cc dd"
    assert overthinking.measure_overthinking_tail(text, tok) == 4
    # No trailing reasoning -> no tail.
    assert overthinking.measure_overthinking_tail("done \\boxed{42}", tok) == 0
    # No answer marker at all -> no tail.
    assert overthinking.measure_overthinking_tail("just words", tok) == 0


def test_compute_overthinking_penalty_is_proportional_and_capped():
    assert overthinking.compute_overthinking_penalty(0, 10, 1.0) == 0.0
    assert overthinking.compute_overthinking_penalty(5, 10, 1.0) == pytest.approx(-0.5)
    # Ratio is capped at 1.0 even if the tail is reported longer than the response.
    assert overthinking.compute_overthinking_penalty(20, 10, 2.0) == pytest.approx(-2.0)


# ── Integration through the (non-new) length_penalty module ──


def _make_experience(torch, response_text, score):
    """Build a single-sample Experience-like object backed by real tensors.

    sequences = [BOS] + response tokens; the WhitespaceTokenizer maps each token id
    back to its string, so token ids ARE the words of ``response_text``.
    """
    words = response_text.split()
    response_ids = list(range(1, len(words) + 1))
    # Decode must reproduce response_text, so override the tokenizer to map id->word.
    sequences = torch.tensor([[0] + response_ids])  # prepend BOS at position 0
    action_mask = torch.ones((1, len(response_ids)), dtype=torch.bool)
    return types.SimpleNamespace(
        sequences=sequences,
        action_mask=action_mask,
        rewards=torch.tensor([1.0]),
        scores=torch.tensor([float(score)]),
        response_length=torch.tensor([len(response_ids)]),
        truncated=None,
        info={"reward": torch.tensor([1.0])},
    ), words


class IdWordTokenizer:
    """Maps the synthetic ids built in _make_experience back to their words."""

    def __init__(self, vocab):
        self.vocab = vocab  # id -> word

    def decode(self, ids, skip_special_tokens=False):
        return " ".join(self.vocab.get(i, "") for i in ids).strip()

    def encode(self, text, add_special_tokens=False):
        return text.split()


def _reward_args(torch):
    return types.SimpleNamespace(
        reward=types.SimpleNamespace(
            overlong_buffer_len=None,
            stop_properly_penalty_coef=None,
            overthinking_penalty_factor=1.0,
            overthinking_min_tail_tokens=1,
        )
    )


def test_apply_length_penalties_penalizes_successful_overthinking():
    torch = pytest.importorskip("torch")

    # Successful trajectory that overthinks: 4 tokens trail the boxed answer.
    overthinker_text = "reason \\boxed{42} aa bb cc dd"
    exp, words = _make_experience(torch, overthinker_text, score=1)
    vocab = {i + 1: w for i, w in enumerate(words)}
    tokenizer = IdWordTokenizer(vocab)

    length_penalty.apply_length_penalties([exp], _reward_args(torch), tokenizer=tokenizer)

    # 4 tail tokens out of 6 response tokens -> reward reduced by ~0.667.
    assert exp.rewards[0].item() == pytest.approx(1.0 - 4 / 6, abs=1e-4)
    # Logged reward is synced with the shaped reward.
    assert exp.info["reward"][0].item() == pytest.approx(exp.rewards[0].item())


def test_apply_length_penalties_spares_unsuccessful_trajectories():
    torch = pytest.importorskip("torch")

    overthinker_text = "reason \\boxed{42} aa bb cc dd"
    exp, words = _make_experience(torch, overthinker_text, score=0)  # unsuccessful
    vocab = {i + 1: w for i, w in enumerate(words)}
    tokenizer = IdWordTokenizer(vocab)

    length_penalty.apply_length_penalties([exp], _reward_args(torch), tokenizer=tokenizer)

    # DRE intervenes only on successful trajectories -> reward untouched.
    assert exp.rewards[0].item() == pytest.approx(1.0)


def test_apply_length_penalties_spares_concise_success():
    torch = pytest.importorskip("torch")

    concise_text = "reason carefully \\boxed{42}"  # no post-answer tail
    exp, words = _make_experience(torch, concise_text, score=1)
    vocab = {i + 1: w for i, w in enumerate(words)}
    tokenizer = IdWordTokenizer(vocab)

    length_penalty.apply_length_penalties([exp], _reward_args(torch), tokenizer=tokenizer)

    assert exp.rewards[0].item() == pytest.approx(1.0)
