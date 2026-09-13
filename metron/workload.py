# Copyright (c) 2024 AXELTEC SOFTWARE LTD.
# SPDX-License-Identifier: Apache-2.0

from transformers import AutoTokenizer
import numpy as np

import random
import json
import os
import re
from pathlib import Path
from abc import ABC, abstractmethod
from typing import List, Optional, Union
from .predefined_datasets import OpenaiRequestsDataset

SEED = int(os.environ.get("SEED", 42))
TOKENIZER = os.environ.get("TOKENIZER", "")
random.seed(SEED)
np.random.seed(SEED)


# Prompts come from a real corpus. The default is GSM8K — grade-school maths
# word problems, MIT licensed, from openai/gsm8k — but any HuggingFace dataset
# or a local JSONL file can be named instead, because the right corpus depends
# on what you are measuring.
#
# What was here before sampled each prompt from a thirteen-word vocabulary:
# "pizza cheese the circle . 12 5 by ( ) : Hello !". Two things break with that.
#
# A small pool repeats, and an engine with prefix caching answers a repeat
# without doing the work. Measured on an RTX 4000: that corpus reported TTFT
# flat at 28 ms from 1 to 16 concurrent, which is not a fast server but a server
# skipping prefill.
#
# And for a mixture-of-experts model it is worse than noise. Expert routing is a
# function of token content, so thirteen tokens keep selecting the same handful
# of experts. The weight movement and load imbalance that dominate MoE serving
# never appear, and the number produced describes a model nobody is running.
CORPUS = os.environ.get("METRON_CORPUS", "openai/gsm8k")
CORPUS_CONFIG = os.environ.get("METRON_CORPUS_CONFIG", "main")
CORPUS_SPLIT = os.environ.get("METRON_CORPUS_SPLIT", "train")
CORPUS_FIELD = os.environ.get("METRON_CORPUS_FIELD", "question")

_corpus: List[str] = []


def _load_corpus() -> List[str]:
    """The text pool, loaded once.

    METRON_CORPUS is either a HuggingFace dataset id or a path to a local
    JSONL file with one object per line. METRON_CORPUS_FIELD names the field
    to read. A corpus that cannot be loaded raises: falling back to synthetic
    text would silently produce the numbers this replaced.
    """
    global _corpus
    if _corpus:
        return _corpus

    path = Path(CORPUS)
    if path.suffix in {".json", ".jsonl"}:
        if not path.is_file():
            raise FileNotFoundError(
                f"METRON_CORPUS points at {path}, which does not exist"
            )
        rows = [
            json.loads(line) for line in path.read_text().splitlines() if line.strip()
        ]
        texts = [str(r.get(CORPUS_FIELD, "")) for r in rows]
    else:
        from datasets import load_dataset

        ds = load_dataset(CORPUS, CORPUS_CONFIG, split=CORPUS_SPLIT)
        if CORPUS_FIELD not in ds.column_names:
            raise KeyError(
                f"{CORPUS} has no field {CORPUS_FIELD!r}; it has {ds.column_names}. "
                f"Set METRON_CORPUS_FIELD."
            )
        texts = [str(t) for t in ds[CORPUS_FIELD]]

    _corpus = [t.strip() for t in texts if t and t.strip()]
    if not _corpus:
        raise ValueError(f"corpus {CORPUS} yielded no usable text")
    random.Random(SEED).shuffle(_corpus)
    return _corpus


def sample_prompt(length: int) -> str:
    """A prompt of roughly `length` words, built from whole corpus entries.

    Entries are taken from a seeded offset and joined until the target length
    is reached, so two prompts share no prefix and an engine cannot answer one
    request by having already served another.
    """
    corpus = _load_corpus()
    start = random.randrange(len(corpus))
    parts: List[str] = []
    words = 0
    i = 0
    while words < length and i < len(corpus):
        entry = corpus[(start + i) % len(corpus)]
        parts.append(entry)
        words += len(entry.split())
        i += 1
    return " ".join(parts)


def limit_prompt_length(prompt: str, length: int, tokenizer: AutoTokenizer) -> str:
    def count_tokens(text: str) -> int:
        return len(tokenizer.tokenize(text))

    current_token_count = count_tokens(prompt)
    assert current_token_count > 0 or length == 0

    if abs(current_token_count - length) == 0:
        return prompt

    while current_token_count < length:
        prompt = prompt + " " + prompt
        current_token_count = count_tokens(prompt)

    max_iterations = 10
    low = 0
    high = len(prompt)
    best_prompt = prompt
    best_diff = abs(current_token_count - length)

    for i in range(max_iterations):
        mid = (low + high) // 2
        truncated_prompt = prompt[:mid]
        token_count = count_tokens(truncated_prompt)
        current_diff = abs(token_count - length)

        if current_diff < best_diff:
            best_diff = current_diff
            best_prompt = truncated_prompt

        if best_diff == 0:
            break

        if token_count < length:
            low = mid
        else:
            high = mid

    return best_prompt


class SequenceGenerator(ABC):
    @abstractmethod
    def get_lengths(self, num_sequences: int) -> List[int]: ...

    def prepare_json(self, path: str, sequence_type: str = "prompts") -> None:
        assert sequence_type in [
            "prompts",
            "generation",
        ], "Please choose `prompts` or `generation` type of sequence"

        if sequence_type == "prompts":
            self.prompts = []

            lengths = self.get_lengths(1000)
            if len(TOKENIZER) >= 1:
                tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
            for length in lengths:
                current_prompt = sample_prompt(length)
                if len(TOKENIZER) >= 1:
                    limited_prompt = limit_prompt_length(
                        current_prompt, length, tokenizer
                    )
                    self.prompts.append(limited_prompt)
                else:
                    self.prompts.append(current_prompt)

            with open(Path(path) / "prompts.json", "w") as file:
                json.dump(self.prompts, file, indent=4)
        else:
            self.lengths = self.get_lengths(1000)

            with open(Path(path) / "gen_lengths.json", "w") as file:
                json.dump(self.lengths, file, indent=4)


class NormallyDistributedSequence(SequenceGenerator):
    def __init__(self, mean: int, std: int) -> None:
        self.mean = mean
        self.std = std

    def get_lengths(self, num_sequences: int) -> List[int]:
        return (
            np.random.normal(loc=self.mean, scale=self.std, size=num_sequences)
            .astype(int)
            .tolist()
        )


class UniformlyDistributedSequence(SequenceGenerator):
    def __init__(self, low: int, high: int) -> None:
        self.low = low
        self.high = high

    def get_lengths(self, num_sequences: int) -> List[int]:
        return (
            np.random.uniform(low=self.low, high=self.high, size=num_sequences)
            .astype(int)
            .tolist()
        )


class ConstantlyDistributedSequence(SequenceGenerator):
    def __init__(self, size: int) -> None:
        self.size = size

    def get_lengths(self, num_sequences: int) -> List[int]:
        return [self.size] * num_sequences


class PredefinedSequence:
    def __init__(
        self,
        path: str,
        context_length: Optional[int] = None,
        min_length: Optional[int] = None,
        truncation_type: str = "end",
    ) -> None:
        self.path = path
        self.context_length = context_length
        self.min_length = min_length
        self.truncation_type = truncation_type
        self.dataset_class = OpenaiRequestsDataset(path)
        if len(TOKENIZER) >= 1:
            self.tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
        else:
            self.tokenizer = None

    def __tokens2symbols_approx(self, num_tokens: int) -> int:
        """
        Usually context length is specified in number of tokens
        but we cannot know the exact number of tokens in text
        because of lack of access to tokenizer in end-to-end
        measurements. So, we approximate the number of symbols
        of text in a given number of tokens.

        We estimate 1 token as ~4 chars
        """
        return int(num_tokens * 4)

    def preprocess_prompts(
        self, prompts: Union[List[str], List[List[int]]]
    ) -> Union[List[str], List[List[int]]]:
        def __escape_control_chars(match):
            char = match.group(0)
            if char in ["\n", "\r", "\t", "\b", "\f"]:
                return f"\\{char}"
            else:
                return ""

        if self.tokenizer is None:
            for prompt_id, text in enumerate(prompts):
                # Remove control characters since it
                # can make JSON invalid
                prompts[prompt_id] = re.sub(
                    r"[\x00-\x1F\x7F]", __escape_control_chars, text
                )  # type: ignore

        if self.min_length is not None:
            assert self.min_length > 0, "min length must be positive"
            if self.tokenizer:
                print("Reducing prompts using tokenizer...")
                min_length = self.min_length
            else:
                print("Reducing prompts using approximate token count...")
                min_length = self.__tokens2symbols_approx(self.min_length)

            prompts = [prompt for prompt in prompts if len(prompt) >= min_length]  # type: ignore

        return prompts

    def truncate_prompts(
        self,
        prompts: Union[List[str], List[List[int]]],
        context_length: Optional[int] = None,
    ) -> List[str]:
        if context_length is None:
            if self.tokenizer:
                prompts = [
                    self.tokenizer.convert_tokens_to_string(prompt)
                    for prompt in prompts
                ]
            return prompts  # type: ignore

        assert context_length > 0, "context length must be positive"

        if self.tokenizer is None:
            print("Truncating prompts using approximate token count...")
            context_length = self.__tokens2symbols_approx(context_length)

        for prompt_id, prompt in enumerate(prompts):
            if self.truncation_type == "end":
                truncated_prompt = prompt[:context_length]
            else:
                truncated_prompt = prompt[-context_length:]
            if self.tokenizer:
                prompts[prompt_id] = self.tokenizer.convert_tokens_to_string(
                    truncated_prompt
                )

        return prompts  # type: ignore

    def prepare_json(self, path: str, sequence_type: str = "prompts") -> None:
        assert sequence_type in [
            "prompts",
            "generation",
        ], "Please choose `prompts` or `generation` type of sequence"

        if sequence_type == "prompts":
            prompts = self.dataset_class.get_prompts(self.tokenizer)
            self.prompts = self.truncate_prompts(
                self.preprocess_prompts(prompts),
                context_length=self.context_length,
            )

            with open(Path(path) / "prompts.json", "w") as file:
                json.dump(self.prompts, file, indent=4)
        else:
            with open(self.path, "r") as file:
                jsonl_content = file.readlines()

            self.lengths = [json.loads(jline)["max_tokens"] for jline in jsonl_content]

            with open(Path(path) / "gen_lengths.json", "w") as file:
                json.dump(self.lengths, file, indent=4)


key2promptgen = {
    "normal": NormallyDistributedSequence,
    "uniform": UniformlyDistributedSequence,
    "const": ConstantlyDistributedSequence,
    "dataset": PredefinedSequence,
}


def arg2generator(arg_str: str) -> SequenceGenerator:
    distribution, parameters = arg_str[:-1].split("(")

    distr_args: List[Union[int, str]] = []
    for arg in parameters.split(","):
        try:
            current_arg: int = int(arg)
            distr_args.append(current_arg)
        except Exception:
            distr_args.append(arg)

    return key2promptgen[distribution](*distr_args)
