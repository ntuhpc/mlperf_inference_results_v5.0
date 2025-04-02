# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.json

from . import *


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class CDI_H100NVLX8(HopperServerGPUBaseConfig):
    system = KnownSystem.H100_NVL_94GBx8
    gpu_batch_size = {'mixtral-8x7b': 896}
    trtllm_build_flags = {
        'max_num_tokens': 16384,
    }
    trtllm_runtime_flags = {'max_num_tokens': 8192}

    #server_target_qps = 250 #OK
    server_target_qps = 260
    #server_target_qps = 300 #NG
    #server_target_qps = 305 #NG
    #server_target_qps = 310 #NG
    #server_target_qps = 325 #NG
    #server_target_qps = 350 #NG




