# Copyright (c) 2024 AXELTEC SOFTWARE LTD.
# SPDX-License-Identifier: Apache-2.0
"""`dataset(...)` must accept exactly the form --help documents.

The second positional used to be a dataset type with one legal value, so the
documented `dataset(file.jsonl, 4096)` raised KeyError: 4096 instead of setting
a context length.
"""

import json

import pytest

from metron.workload import arg2generator


@pytest.fixture
def prompts(tmp_path):
    p = tmp_path / "prompts.jsonl"
    p.write_text(
        "".join(json.dumps({"content": f"prompt {i}"}) + "\n" for i in range(4))
    )
    return p


def test_path_only(prompts):
    gen = arg2generator(f"dataset({prompts})")
    assert gen.context_length is None


def test_path_and_context_length(prompts):
    gen = arg2generator(f"dataset({prompts}, 4096)")
    assert gen.context_length == 4096


def test_context_length_is_not_swallowed_as_a_type(prompts):
    """The regression itself: 4096 must not be read as a dataset type."""
    gen = arg2generator(f"dataset({prompts}, 4096)")
    assert not isinstance(gen.context_length, str)
