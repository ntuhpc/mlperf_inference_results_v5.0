# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class ARS_111GL_NHR(HopperServerGPUBaseConfig):
    system = KnownSystem.ARS_111GL_NHR

    gpu_batch_size = {'llama2-70b': 2048}
    server_target_qps = 10.5
    trtllm_build_flags = {
        'max_num_tokens': 1536, 
        'gemm_swiglu_plugin': 'fp8',
    }
    trtllm_runtime_flags = {'max_num_tokens': 1536}

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class ARS_111GL_NHR_HighAccuracy(ARS_111GL_NHR):
    pass
