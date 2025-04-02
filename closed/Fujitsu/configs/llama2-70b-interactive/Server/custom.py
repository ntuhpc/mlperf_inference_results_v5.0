
from . import *

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP, "TP2")
class CDI_H100NVLX8(HopperServerGPUBaseConfig):
    system = KnownSystem.H100_NVL_94GBx8
    gpu_batch_size = {'llama2-70b-interactive': 512}
    #offline_expected_qps = 50
    #server_target_qps = 35 #OK
    #server_target_qps = 43 #NG 
    #server_target_qps = 39 #NG
    server_target_qps = 38.5
    #server_target_qps = 41 #NG
    #server_target_qps = 40 #NG
    trtllm_runtime_flags = {'max_num_tokens': 256}
    trtllm_build_flags = {
        'max_num_tokens': 256,
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
