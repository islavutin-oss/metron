# Copyright (c) 2024 AXELTEC SOFTWARE LTD.
# SPDX-License-Identifier: Apache-2.0

import pytest
from transformers import AutoTokenizer
from metron.workload import limit_prompt_length

TOKENIZER_NAME = "bert-base-uncased"


@pytest.fixture
def tokenizer():
    return AutoTokenizer.from_pretrained(TOKENIZER_NAME)


@pytest.mark.parametrize(
    "prompt,target_length",
    [
        ("The quick brown fox jumps over the lazy dog", 5),
        ("Lorem ipsum dolor sit amet, consectetur adipiscing elit.", 10),
        (
            "A very very long sentence that keeps going on and on and might need trimming.",
            8,
        ),
        ("Machine learning models can be very complex", 4),
        ("Short text", 2),
        ("Short", 2),
        ("A" * 100, 10),
    ],
)
def test_limit_prompt_length(prompt, target_length, tokenizer):
    result = limit_prompt_length(prompt, target_length, tokenizer)
    tokens = tokenizer.tokenize(result)

    assert abs(len(tokens) - target_length) == 0
    assert isinstance(result, str)


def test_exact_length(tokenizer):
    prompt = "This is a test prompt."
    tokens = tokenizer.tokenize(prompt)
    result = limit_prompt_length(prompt, len(tokens), tokenizer)
    assert result == prompt


def test_special_chars(tokenizer):
    prompt = "(Hello) @world!"
    result = limit_prompt_length(prompt, 4, tokenizer)
    assert len(tokenizer.tokenize(result)) <= 4


def test_truncation_accuracy(tokenizer):
    prompt = "Repeat " * 100
    max_tokens = 20
    result = limit_prompt_length(prompt, max_tokens, tokenizer)
    token_count = len(tokenizer.tokenize(result))
    assert token_count <= max_tokens
