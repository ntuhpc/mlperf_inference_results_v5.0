# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.json

from . import *


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class SR675V3_H200_SXMX4(HopperServerGPUBaseConfig):
    system = KnownSystem.SR675v3_H200_SXMx4
    gpu_batch_size = {'llama2-70b': 2048}
    server_target_qps = 50.8
    trtllm_build_flags = {
        'max_num_tokens': 1536,
        'gemm_swiglu_plugin': 'fp8',
    }
    trtllm_runtime_flags = {'max_num_tokens': 1536}


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class SR675V3_H200_SXMX4_HighAccuracy(SR675V3_H200_SXMX4):
    pass


