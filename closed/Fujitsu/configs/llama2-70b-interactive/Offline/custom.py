
from . import *

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP, "TP2")
class CDI_H100NVLX8(HopperOfflineGPUBaseConfig):
    system = KnownSystem.H100_NVL_94GBx8
    gpu_batch_size = {'llama2-70b-interactive': 540}
    #offline_expected_qps = 50
    offline_expected_qps = 55
    trtllm_runtime_flags = {'max_num_tokens': 512}
    trtllm_build_flags = {
        'max_num_tokens': 512,
        'tensor_parallelism': 2,
        'pipeline_parallelism': 1,
    }

    #gpu_batch_size = {'llama2-70b-interactive': 2048}
    #trtllm_build_flags = {
    #    'max_num_tokens': 1536,
    #    'gemm_swiglu_plugin': 'fp8',
    #}
    #trtllm_runtime_flags = {'max_num_tokens': 1536}
    #offline_expected_qps = 14.4

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP, "TP2")
class CDI_H100NVLX8_HighAccuracy(CDI_H100NVLX8):
    pass
