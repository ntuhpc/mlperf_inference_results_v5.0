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

ParentConfig = import_module("configs.llama2-70b")
GPUBaseConfig = ParentConfig.GPUBaseConfig


class ServerGPUBaseConfig(GPUBaseConfig):
    scenario = Scenario.Server
    min_duration = 2400000
    enable_sort = False


class HopperServerGPUBaseConfig(ServerGPUBaseConfig):
    precision = "fp8"
    vboost_slider = 1

    trtllm_checkpoint_flags = {
        'kv_cache_dtype': 'fp8'
    }

    trtllm_build_flags = {
        'tensor_parallelism': 1,
        'pipeline_parallelism': 1,
    }


class BlackwellServerGPUBaseConfig(ServerGPUBaseConfig):
    precision = 'fp4'

    trtllm_checkpoint_flags = {
        'kv_cache_dtype': 'fp8'
    }

    trtllm_build_flags = {
        'tensor_parallelism': 1,
        'pipeline_parallelism': 1,
        'norm_quant_fusion': 'enable'
    }


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP, "TP2")
class D54U_3U_H100_PCIe_80GBx4(HopperServerGPUBaseConfig):
    system = KnownSystem.D54U_3U_H100_PCIe_80GBx4
    gpu_batch_size = {'llama2-70b': 2048}
    server_target_qps = 17

    trtllm_build_flags = {
        'tensor_parallelism': 2,
        'pipeline_parallelism': 1,
        'gemm_swiglu_plugin': 'fp8',
    }


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class D54U_3U_H100_PCIe_80GBx4_HighAccuracy(D54U_3U_H100_PCIe_80GBx4):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP, "PP2")
class D74U_7U_H100_SXM_80GBx8(HopperServerGPUBaseConfig):
    system = KnownSystem.H100_SXM_80GBx8
    server_target_qps = 103
    vboost_slider = 0
    min_duration = 3600000

    gpu_batch_size = {'llama2-70b': 1024}
    trtllm_build_flags = {
        'max_num_tokens': 1024,
        'tensor_parallelism': 1,
        'pipeline_parallelism': 2,
        'reduce_fusion': 'enable',
        'gemm_swiglu_plugin': 'fp8',
    }
    trtllm_runtime_flags = {
        'max_num_tokens': 1024,
        'kvcache_free_gpu_mem_frac': 0.95,
    }


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP, "PP2")
class D74U_7U_H100_SXM_80GBx8_HighAccuracy(D74U_7U_H100_SXM_80GBx8):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class S74G_2U_GH200_96GB_aarch64x1(HopperServerGPUBaseConfig):
    system = KnownSystem.GH200_96GB_ARMx1

    gpu_batch_size = {'llama2-70b': 2048}
    server_target_qps = 11
    trtllm_build_flags = {
        'max_num_tokens': 1536,
        'gemm_swiglu_plugin': 'fp8',
    }
    trtllm_runtime_flags = {'max_num_tokens': 1536}


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class S74G_2U_GH200_96GB_aarch64x1_HighAccuracy(S74G_2U_GH200_96GB_aarch64x1):
    pass



