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
from code.common.constants import Benchmark, Scenario
from code.common.systems.system_list import KnownSystem
from configs.configuration import *

ParentConfig = import_module("configs.dlrm-v2")
GPUBaseConfig = ParentConfig.GPUBaseConfig


class ServerGPUBaseConfig(GPUBaseConfig):
    scenario = Scenario.Server


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class D54U_3U_H100_PCIe_80GBx4(ServerGPUBaseConfig):
    system = KnownSystem.D54U_3U_H100_PCIe_80GBx4
    gpu_batch_size = {'dlrm-v2': 51200}
    embedding_weights_on_gpu_part: float = 1.0
    server_target_qps = 175400
    numa_config = "0-1:0-51,104-155&2-3:52-103,156-207"


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class D54U_3U_H100_PCIe_80GBx4_HighAccuracy(D54U_3U_H100_PCIe_80GBx4):
    server_target_qps = 100200
    interaction_op_precision = 'fp16'


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class D74U_7U_H100_SXM_80GBx8(ServerGPUBaseConfig):
    system = KnownSystem.H100_SXM_80GBx8
    gpu_batch_size = {'dlrm-v2': 51200}
    embedding_weights_on_gpu_part: float = 1.0
    server_target_qps = 512000
    server_num_issue_query_threads = 8
    vboost_slider = 1
    numa_config = "0:0-13,112-125&1:28-41,140-153&2:42-55,154-167&3:14-27,126-139&4:56-69,168-181&5:84-97,196-209&6:98-111,210-223&7:70-83,182-195"


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class D74U_7U_H100_SXM_80GBx8_HighAccuracy(D74U_7U_H100_SXM_80GBx8):
    server_target_qps = 340000
    interaction_op_precision = 'fp16'


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class S74G_2U_GH200_96GB_aarch64x1(ServerGPUBaseConfig):
    system = KnownSystem.GH200_96GB_ARMx1
    gpu_batch_size = {'dlrm-v2': 204800}
    embedding_weights_on_gpu_part: float = 1.0
    server_target_qps = 84000


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class S74G_2U_GH200_96GB_aarch64x1_High_Accuracy(S74G_2U_GH200_96GB_aarch64x1):
    server_target_qps = 51000
    interaction_op_precision = 'fp16'


