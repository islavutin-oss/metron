# Copyright (c) 2024 AXELTEC SOFTWARE LTD.
# SPDX-License-Identifier: Apache-2.0

"""Prompts must be real text, and must not repeat.

The default corpus used to be thirteen words sampled at random, every prompt
prefixed with the same literal. Two things broke.

An engine with prefix caching answers a repeated prompt without doing the work.
Measured on an RTX 4000: that corpus reported TTFT flat at 28 ms from 1 to 16
concurrent users, which is not a fast server but a server skipping prefill.
With caching off the same sweep climbed 27 to 90 ms.

And on a mixture-of-experts model it is worse than noise: routing is a function
of token content, so a thirteen-token vocabulary keeps selecting the same few
experts and the weight movement that dominates MoE serving never appears.
"""

import importlib
import json

import pytest


def _workload(monkeypatch, **env):
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    import metron.workload as w

    importlib.reload(w)
    return w


@pytest.fixture
def corpus_file(tmp_path):
    p = tmp_path / "corpus.jsonl"
    p.write_text(
        "\n".join(
            json.dumps(
                {
                    "question": f"Problem {i}. "
                    + " ".join(f"w{i}x{j}" for j in range(40))
                }
            )
            for i in range(200)
        )
    )
    return p


def test_a_local_corpus_can_be_supplied(monkeypatch, corpus_file):
    w = _workload(monkeypatch, METRON_CORPUS=str(corpus_file))
    assert w.sample_prompt(20).startswith("Problem ")


def test_prompts_do_not_share_a_prefix(monkeypatch, corpus_file):
    """The property that makes prefix caching irrelevant to the measurement."""
    w = _workload(monkeypatch, METRON_CORPUS=str(corpus_file))
    heads = {w.sample_prompt(30)[:60] for _ in range(50)}
    assert len(heads) > 40, f"only {len(heads)} distinct prefixes in 50 prompts"


def test_a_prompt_reaches_the_requested_length(monkeypatch, corpus_file):
    w = _workload(monkeypatch, METRON_CORPUS=str(corpus_file))
    assert len(w.sample_prompt(120).split()) >= 120


def test_a_missing_corpus_fails_loudly(monkeypatch, tmp_path):
    """Falling back to synthetic text would silently restore the old numbers."""
    w = _workload(monkeypatch, METRON_CORPUS=str(tmp_path / "nope.jsonl"))
    with pytest.raises(FileNotFoundError):
        w.sample_prompt(10)


def test_an_empty_corpus_fails_loudly(monkeypatch, tmp_path):
    empty = tmp_path / "empty.jsonl"
    empty.write_text(json.dumps({"question": "   "}) + "\n")
    w = _workload(monkeypatch, METRON_CORPUS=str(empty))
    with pytest.raises(ValueError):
        w.sample_prompt(10)


def test_a_wrong_field_name_says_so(monkeypatch, corpus_file):
    w = _workload(
        monkeypatch, METRON_CORPUS=str(corpus_file), METRON_CORPUS_FIELD="not_a_field"
    )
    with pytest.raises(ValueError):
        w.sample_prompt(10)


def test_the_synthetic_vocabulary_is_gone():
    """It was thirteen words; nothing should reintroduce a fixed token list."""
    import pathlib

    src = (
        pathlib.Path(__file__).resolve().parents[1] / "metron" / "workload.py"
    ).read_text()
    body = src.split("def sample_prompt", 1)[1].split("\ndef ", 1)[0]
    assert '"pizza"' not in src
    assert "random.choices" not in body, "prompts are sampled from a token list again"


def test_the_default_corpus_is_named_and_overridable():
    import metron.workload as w

    importlib.reload(w)
    assert w.CORPUS == "openai/gsm8k"
