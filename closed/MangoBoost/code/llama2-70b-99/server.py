# Copyright 2024, MangoBoost, Inc. All rights reserved.

import logging
import os
from contextlib import asynccontextmanager

from llmboost import LLMBoost

from absl import app, flags
from llmboost.mlperf_server_func import llmboost_mlperf_server, set_model_map

model_map = {
    "llama2-70b": "/models/amd2025_model/model/llama2-70b-chat-hf/fp8_quantized",
}

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("MB-MLPERF-INFERENCE-SERVER")

FLAGS = flags.FLAGS

flags.DEFINE_enum("model_name", "llama2-70b", model_map.keys(), "Model name")
flags.DEFINE_string("host", None, "Host address")
flags.DEFINE_integer("port", 8000, "Port number")
flags.DEFINE_integer("max_tokens", 1024, "max tokens to generate")
flags.DEFINE_integer("tp", 1, "tensor parallelism")
flags.DEFINE_integer("dp", 8, "number of workers")
flags.DEFINE_integer("beam_width", 0, "beam width if used")
flags.DEFINE_enum(
    "load_format",
    "auto",
    ["bitsandbytes", "bitsandbytesint8", "auto"],
    "Quantized model to use",
)
flags.DEFINE_enum(
    "quantization",
    "None",
    ["bitsandbytes", "fp8", "None"],
    "Quantization scheme to use",
)
flags.DEFINE_enum("kv_cache_dtype", "auto", ["auto", "fp8"], "KV cache dtype")
flags.DEFINE_string("quantization_param_path", None, "Path to quantization parameters")
flags.DEFINE_string("quantization_weight_path", None, "Path to quantization weights")
flags.DEFINE_integer(
    "max_num_seqs", 0, "max number of sequences to process in parallel"
)

flags.register_validator(
    "tp", lambda x: x > 0 and x < 9, message="tp must be between 1 and 8"
)
flags.register_validator(
    "dp", lambda x: x > 0 and x < 9, message="dp must be between 1 and 8"
)
flags.register_validator(
    "max_num_seqs", lambda x: x > 0, message="max_num_seqs must be greater than 0"
)
flags.DEFINE_enum(
    "test_mode", "Offline", ["Offline", "Server"], "type of test to perform"
)
flags.DEFINE_string("load_balancing_mode", "auto", "Load Balancing method")
flags.DEFINE_integer(
    "gpu_batch_size",
    48,
    "The max number of queries to be sent to each worker as a batch.",
)
flags.DEFINE_float(
    "batcher_threshold",
    0.4,
    "The max duration (in sec) that we wait to send the pending queries.",
)
flags.DEFINE_bool(
    "output_semi_batching",
    False,
    "Per query only send the outputs twice (first token and rest tokerns)",
)


def main(argv):
    del argv
    set_model_map(model_map)

    llmboost_mlperf_server(flags = FLAGS)


if __name__ == "__main__":
    app.run(main)
