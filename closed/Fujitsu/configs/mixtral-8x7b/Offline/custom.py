# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.json

from . import *


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class CDI_H100NVLX8(HopperOfflineGPUBaseConfig):
    system = KnownSystem.H100_NVL_94GBx8
    
    gpu_batch_size = {'mixtral-8x7b': 896}
    trtllm_build_flags = {'max_num_tokens': 8192}
    trtllm_runtime_flags = {'max_num_tokens': 8192}

    #offline_expected_qps = 46 * 8
    offline_expected_qps = 260

