# Copyright (c) 2024 AXELTEC SOFTWARE LTD.
# SPDX-License-Identifier: Apache-2.0

import abc
import json
from typing import List, Optional, Union
from transformers import AutoTokenizer
import os
import pickle
from pathlib import Path

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_TOKENIZED_DIR = Path(os.path.join(CURRENT_DIR, "tokenized_datasets"))


class BaseDataset(abc.ABC):
    def __init__(self, path: str, tokenized_dataset_name: str = "") -> None:
        self.path = path
        self.tokenized_dataset_name = tokenized_dataset_name

    @abc.abstractmethod
    def prepare_prompts(self) -> List[str]:
        """Get prompts from the dataset."""
        raise NotImplementedError("Subclasses should implement this method.")

    def tokenize_prompts(
        self, prompts: List[str], tokenizer: AutoTokenizer
    ) -> List[List[int]]:
        assert tokenizer is not None

        tokenizer_name = tokenizer.name_or_path.split("/")[-1]
        path_to_tokenized_datasets = DEFAULT_TOKENIZED_DIR.joinpath(tokenizer_name)
        os.makedirs(path_to_tokenized_datasets, exist_ok=True)
        path_to_dataset = path_to_tokenized_datasets.joinpath(
            self.tokenized_dataset_name
        )

        if os.path.exists(path_to_dataset):
            print("Loading tokenized dataset from disk...")
            with open(path_to_dataset, "rb") as f:
                tokenized_prompts = pickle.load(f)
        else:
            print("Starting tokenization...")
            tokenized_prompts = [tokenizer.tokenize(prompt) for prompt in prompts]
            print("Tokenization is over")
            with open(path_to_dataset, "wb") as f:
                pickle.dump(tokenized_prompts, f)

        return tokenized_prompts

    def get_prompts(
        self, tokenizer: Optional[AutoTokenizer] = None
    ) -> Union[List[str], List[List[int]]]:
        """Get prompts from the dataset."""
        prompts = self.prepare_prompts()
        if tokenizer:
            prompts = self.tokenize_prompts(prompts, tokenizer)  # type: ignore

        return prompts


class OpenaiRequestsDataset(BaseDataset):
    def __init__(self, path: str) -> None:
        super().__init__(
            path=path,
            tokenized_dataset_name="openai_requests_tokenized.pickle",
        )

    def prepare_prompts(self) -> List[str]:
        if self.path.endswith("json"):
            try:
                with open(self.path, "r") as file:
                    json_content = json.load(file)
            except FileNotFoundError:
                raise FileNotFoundError(f"File {self.path} not found.")

            prompts = [elem["prompt"] for elem in json_content]
        elif self.path.endswith("jsonl"):
            try:
                with open(self.path, "r") as file:
                    jsonl_content = file.readlines()
            except FileNotFoundError:
                raise FileNotFoundError(f"File {self.path} not found.")

            prompts = [json.loads(jline)["content"] for jline in jsonl_content]
        return prompts
