# Copyright 2024, MangoBoost, Inc. All rights reserved.

import logging
import os

# import ABC
from abc import ABC, abstractmethod

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("DATASET")


class Dataset(ABC):
    def __init__(
        self, model_name, dataset_path, total_sample_count, input_key, output_key
    ) -> None:
        self.model_name = model_name
        self.dataset_path = dataset_path
        self.total_sample_count = total_sample_count
        self.perf_count = total_sample_count

        # Load the tokenizer
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            model_max_length=2048,
            padding_side="left",
            use_fast=False,
            trust_remote_code=True,
        )

        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.load_data()

    @abstractmethod
    def load_data(self):
        raise NotImplementedError

    def LoadSamplesToRam(self, sample_list):
        pass

    def UnloadSamplesFromRam(self, sample_list):
        pass


class GPTJDataset(Dataset):
    PROMPT_INPUT = (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    )

    def load_data(self):
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Dataset file not found at {self.dataset_path}")

        log.debug(f"Loading dataset from {self.dataset_path}")

        # Read in the prompts
        import json
        import io

        f = self.dataset_path
        if not isinstance(f, io.IOBase):
            f = open(f, "r")
        jdict = json.load(f)
        f.close()

        # Format them
        prompts = [self.PROMPT_INPUT.format_map(example) for example in jdict]

        # Encode them
        self.input_tokens = {}
        for i, prompt in enumerate(prompts):
            prompt_tokens = (
                self.tokenizer(
                    prompt,
                    return_tensors="np",
                    padding=True,
                    truncation=True,
                    max_length=1919,
                )
                .input_ids[0]
                .tolist()
            )
            self.input_tokens[i] = prompt_tokens
        log.debug("Dataset loaded")


class MixtralDataset(Dataset):
    def load_data(self):
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Dataset file not found at {self.dataset_path}")

        log.debug(f"Loading dataset from {self.dataset_path}")
        import pandas as pd

        processed_data = pd.read_pickle(self.dataset_path)

        self.input_tokens = {}
        for i, ids in enumerate(processed_data["tok_input"]):
            self.input_tokens[i] = ids
        log.debug("Dataset loaded")


class LlamaDataset(Dataset):
    def load_data(self):
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Dataset file not found at {self.dataset_path}")

        log.debug(f"Loading dataset from {self.dataset_path}")
        import pandas as pd

        processed_data = pd.read_pickle(self.dataset_path)

        self.input_tokens = {}
        for i, ids in enumerate(processed_data["tok_input"]):
            self.input_tokens[i] = ids

        log.debug("Dataset loaded")


def get_dataset_info(model_name):
    # if the model has llama in it, then it will be using the llama dataset
    model_name = model_name.lower()
    dataset = None
    if "llama" in model_name:
        dataset_info = dataset_info_map["llama-openorca"]
        dataset = dataset_map["llama-openorca"](**dataset_info)
    elif "gptj" in model_name:
        dataset_info = dataset_info_map["gptj-dataset"]
        dataset = dataset_map["gptj"](**dataset_info)
    elif "mixtral" in model_name:
        dataset_info = dataset_info_map["mixtral-dataset"]
        dataset = dataset_map["mixtral-dataset"](**dataset_info)
    else:
        raise ValueError(f"Model name {model_name} not found in dataset_info_map")

    return dataset


dataset_map = {
    "gptj": GPTJDataset,
    "llama-openorca": LlamaDataset,
    "mixtral-dataset": MixtralDataset,
}

dataset_info_map = {
    "gptj-dataset": {
        "model_name": "/workspace/models/gpt-j-6b",
        "dataset_path": "/workspace/data/gptj_data.json",
        "total_sample_count": 13368,
    },
    "llama-openorca": {
        "model_name": "/models/amd2025_model/model/llama2-70b-chat-hf/fp8_quantized",
        "dataset_path": "/models/amd2025_model/data/processed-openorca/open_orca_gpt4_tokenized_llama.sampled_24576.pkl",
        "total_sample_count": 24576,
        "input_key": "tok_input",
        "output_key": "output",
    },
    "mixtral-dataset": {
        "model_name": "/workspace/models/mixtral-8x7b-instruct-v0.1",
        "dataset_path": "/workspace/data/mixtral_data.pkl",
        "total_sample_count": 15000,
        "input_key": "tok_input",
        "output_key": "ref_output",
    },
}
