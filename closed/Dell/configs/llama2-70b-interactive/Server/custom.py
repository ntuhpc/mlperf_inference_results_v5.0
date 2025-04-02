from . import *

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE9680L_H200_SXM_141GBX8(HopperServerGPUBaseConfig):
    system = KnownSystem.XE9680L_H200_SXM_141GBx8
    gpu_batch_size = {'llama2-70b-interactive': 512}
    trtllm_build_flags = {'max_num_tokens': 256}
    trtllm_runtime_flags = {'max_num_tokens': 256}
    server_target_qps = 8.4 * 8

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class XE9680L_H200_SXM_141GBX8_HighAccuracy(XE9680L_H200_SXM_141GBX8):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE9680_H200_SXM_141GBX8(ServerGPUBaseConfig):
    system = KnownSystem.XE9680_H200_SXM_141GBx8
    gpu_batch_size = {'llama2-70b-interactive': 512}
    server_target_qps = 8.4 * 8
    precision = 'fp8'
    vboost_slider = 1
    trtllm_build_flags = {
       'max_num_tokens': 256,
       'tensor_parallelism': 1,
       'pipeline_parallelism': 1,
    }
    trtllm_checkpoint_flags = {
        'kv_cache_dtype': 'fp8',
    }
    trtllm_runtime_flags = {
        'kvcache_free_gpu_mem_frac': 0.90,
        'max_num_tokens': 256,
    }

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class XE9680_H200_SXM_141GBX8_HighAccuracy(XE9680_H200_SXM_141GBX8):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE9680_H100_SXM_80GBX8(HopperServerGPUBaseConfig):
    system = KnownSystem.XE9680_H100_SXM_80GBx8
    trtllm_build_flags = {
        'tensor_parallelism': 1,
        'pipeline_parallelism': 2,
    }
    gpu_batch_size = {'llama2-70b-interactive': 512}
    trtllm_build_flags = {'max_num_tokens': 256}
    trtllm_runtime_flags = {'max_num_tokens': 256}
    server_target_qps = 8.4 * 8

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class XE9680_H100_SXM_80GBX8_HighAccuracy(XE9680_H100_SXM_80GBX8):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE7745_H200_NVL_141GBX8(HopperServerGPUBaseConfig):
    system = KnownSystem.XE7745_H200_NVL_141GBx8
    gpu_batch_size = {'llama2-70b-interactive': 512}
    trtllm_build_flags = {
            'max_num_tokens': 256,
            'gemm_swiglu_plugin': 'fp8',}
    trtllm_runtime_flags = {'max_num_tokens': 256}
    server_target_qps = 56

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class XE7745_H200_NVL_141GBX8_HighAccuracy(XE7745_H200_NVL_141GBX8):
    pass

