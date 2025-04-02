# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
sys.path.insert(0, os.getcwd())

from importlib import import_module
from code.common.constants import Scenario
from code.common.systems.system_list import KnownSystem
from configs.configuration import *

ParentConfig = import_module("configs.mixtral-8x7b")
GPUBaseConfig = ParentConfig.GPUBaseConfig


class ServerGPUBaseConfig(GPUBaseConfig):
    scenario = Scenario.Server
    enable_sort = False
    min_duration = 2400000

    trtllm_runtime_flags = {
        'batch_scheduler_policy': 'max_util',
        'context_chunking_policy': 'first_come_first_served',
    }


class HopperServerGPUBaseConfig(ServerGPUBaseConfig):
    precision = 'fp8'
    vboost_slider = 1

    trtllm_runtime_flags = {
        'kvcache_free_gpu_mem_frac': 0.90,
        'enable_chunked_context': True,
    }

    trtllm_checkpoint_flags = {
        'kv_cache_dtype': 'fp8',
        'effective_bits': 8.75,
        'num_calib_steps': 16,
    }

    trtllm_build_flags = {
        'tokens_per_block': 32,
        'tensor_parallelism': 1,
        'pipeline_parallelism': 1,
    }


class BlackwellServerGPUBaseConfig(ServerGPUBaseConfig):
    precision = "fp4"
    # vboost_slider = 1

    trtllm_build_flags = {
        # 'use_paged_context_fmha': 'enable',
        'tokens_per_block': 32,
        'use_fp8_context_fmha': 'enable',
        'gemm_plugin': 'fp4',
        'tensor_parallelism': 1,
        'pipeline_parallelism': 1,
    }

    trtllm_checkpoint_flags = {
        'kv_cache_dtype': 'fp8',
        'effective_bits': 7.25,
        'num_score_steps': 8,
        'num_calib_steps': 64,
    }

    trtllm_runtime_flags = {
        'kvcache_free_gpu_mem_frac': 0.90,
        # 'enable_chunked_context': True,
    }


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class D54U_3U_H100_PCIe_80GBx4(HopperServerGPUBaseConfig):
    system = KnownSystem.D54U_3U_H100_PCIe_80GBx4
    gpu_batch_size = {'mixtral-8x7b': 896}
    server_target_qps = 92


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class D74U_7U_H100_SXM_80GBx8(HopperServerGPUBaseConfig):
    system = KnownSystem.H100_SXM_80GBx8
    gpu_batch_size = {'mixtral-8x7b': 896}
    server_target_qps = 346


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class S74G_2U_GH200_96GB_aarch64x1(HopperServerGPUBaseConfig):
    system = KnownSystem.GH200_96GB_ARMx1
    gpu_batch_size = {'mixtral-8x7b': 4096}
    server_target_qps = 53
    trtllm_build_flags = {'max_num_tokens': 10 * 1024}
    trtllm_runtime_flags = {
        'max_num_tokens': 9 * 1024,
    }


