# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class HPE_PROLIANT_DL380A_H200_NVL_141GBX8(HopperOfflineGPUBaseConfig):
    system = KnownSystem.HPE_PROLIANT_DL380A_H200_NVL_141GBX8
    offline_expected_qps = 0.67
    trtllm_build_flags = {
        'max_num_tokens': 2560,
        'tensor_parallelism': 4,
        'pipeline_parallelism': 2,
    }
    trtllm_runtime_flags = {'max_num_tokens': 2560}
    gpu_batch_size = {'llama3.1-405b': 512}
