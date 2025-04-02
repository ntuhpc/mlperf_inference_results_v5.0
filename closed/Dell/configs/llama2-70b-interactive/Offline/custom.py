from . import *

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE9680L_H200_SXM_141GBX8(HopperOfflineGPUBaseConfig):
    system = KnownSystem.XE9680L_H200_SXM_141GBx8
    gpu_batch_size = {'llama2-70b-interactive': 2048}
    trtllm_build_flags = {
        'max_num_tokens': 1536,
        'gemm_swiglu_plugin': 'fp8',
    }
    trtllm_runtime_flags = {'max_num_tokens': 1536}
    offline_expected_qps = 14.4 * 8

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class XE9680L_H200_SXM_141GBX8_HighAccuracy(XE9680L_H200_SXM_141GBX8):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE9680_H200_SXM_141GBX8(OfflineGPUBaseConfig):
    system = KnownSystem.XE9680_H200_SXM_141GBx8
    gpu_batch_size = {'llama2-70b-interactive': 2048}
    precision = "fp8"
    vboost_slider = 1
    trtllm_runtime_flags = {
        'kvcache_free_gpu_mem_frac': 0.90,
        'enable_chunked_context': True,
        'max_num_tokens': 1536
    }
    trtllm_checkpoint_flags = {
        'kv_cache_dtype': 'fp8'
    }
    trtllm_build_flags = {
        'use_paged_context_fmha': 'enable',
        'tokens_per_block': 32,
        'tensor_parallelism': 1,
        'pipeline_parallelism': 1,
        'max_num_tokens': 1536,
        'gemm_swiglu_plugin': 'fp8'
    }
    offline_expected_qps = 14.4 * 8

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class XE9680_H200_SXM_141GBX8_HighAccuracy(XE9680_H200_SXM_141GBX8):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE9680_H100_SXM_80GBX8(HopperOfflineGPUBaseConfig):
    system = KnownSystem.XE9680_H100_SXM_80GBx8
    vboost_slider = 0
    min_duration = 3600000
    gpu_batch_size = {'llama2-70b': 1024}
    offline_expected_qps = 27.5 * 4
    trtllm_build_flags = {
        'max_num_tokens': 1024,
        'tensor_parallelism': 1,
        'pipeline_parallelism': 2,
        'reduce_fusion': 'enable',
        'gemm_swiglu_plugin': 'fp8',
    }
    trtllm_runtime_flags = {
        'max_num_tokens': 1024,
        'kvcache_free_gpu_mem_frac': 0.95,
    }

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class XE9680_H100_SXM_80GBX8_HighAccuracy(XE9680_H100_SXM_80GBX8):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE7745_H200_NVL_141GBx8(HopperOfflineGPUBaseConfig):
    system = KnownSystem.XE7745_H200_NVL_141GBx8
    gpu_batch_size = {'llama2-70b-interactive': 2048}
    trtllm_build_flags = {
        'max_num_tokens': 1536,
        'gemm_swiglu_plugin': 'fp8',
    }
    trtllm_runtime_flags = {'max_num_tokens': 1536}
    offline_expected_qps = 115.2

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class XE7745_H200_NVL_141GBX8_HighAccuracy(XE7745_H200_NVL_141GBx8):
    pass

