from . import *

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP, "TP8PP1")
class HPE_PROLIANT_DL380A_H200_NVL_141GBX8(HopperServerGPUBaseConfig):
    system = KnownSystem.HPE_PROLIANT_DL380A_H200_NVL_141GBX8

    gpu_batch_size = {'llama3.1-405b': 512}
    trtllm_build_flags = {
        'max_num_tokens': 8192,
        'tensor_parallelism': 8,
        'pipeline_parallelism': 1,
        'gemm_allreduce_plugin': 'float16',
    }
    trtllm_runtime_flags = {
        'max_num_tokens': 2560,
        'max_batch_size': 64,
        'kvcache_free_gpu_mem_frac': 0.9
    }
    server_target_qps = 0.45

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP, "TP8PP1")
class HPE_PROLIANT_DL380A_H200_NVL_141GBX8_HighAccuracy(HPE_PROLIANT_DL380A_H200_NVL_141GBX8):
    pass

"""
@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP, "TP4PP2")
class H200_SXM_141GBx8_TP4PP2(HopperOfflineGPUBaseConfig):
    system = KnownSystem.H200_SXM_141GBx8

    gpu_batch_size = {'llama3.1-405b': 2 * 1024}
    trtllm_build_flags = {
        'max_num_tokens': 2560,
        'tensor_parallelism': 4,
        'pipeline_parallelism': 2,
    }
    trtllm_runtime_flags = {'max_num_tokens': 2560}
    server_target_qps = 0.5


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP, "TP8PP1")
class H200_SXM_141GBx8_TP8PP1(HopperOfflineGPUBaseConfig):
    system = KnownSystem.H200_SXM_141GBx8

    gpu_batch_size = {'llama3.1-405b': 2 * 1024}
    trtllm_build_flags = {
        'max_num_tokens': 2560,
        'tensor_parallelism': 8,
        'pipeline_parallelism': 1,
    }
    trtllm_runtime_flags = {'max_num_tokens': 2560}
    server_target_qps = 0.5
"""
