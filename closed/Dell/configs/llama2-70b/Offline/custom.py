from . import *


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE7745_L40SX8(OfflineGPUBaseConfig):
    system = KnownSystem.XE7745_L40Sx8
    gpu_batch_size = {'llama2-70b': 1024}
    precision="fp8"
    offline_expected_qps = 15
    enable_sort = False
    trtllm_build_flags = {
        'tensor_parallelism': 2,
        'pipeline_parallelism': 1,
    }

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class XE7745_L40SX8_HighAccuracy(XE7745_L40SX8):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE9680L_H200_SXM_141GBX8(HopperOfflineGPUBaseConfig):
    system = KnownSystem.XE9680L_H200_SXM_141GBx8
    gpu_batch_size = {'llama2-70b': 2048}
    offline_expected_qps = 14.4 * 8
    precision = "fp8"
    vboost_slider = 1
    trtllm_checkpoint_flags = {
        'kv_cache_dtype': 'fp8'
    }
    trtllm_build_flags = {
        'tensor_parallelism': 1,
        'pipeline_parallelism': 1,
        'max_num_tokens': 1536,
        'gemm_swiglu_plugin': 'fp8',
    }
    trtllm_runtime_flags = {'max_num_tokens': 1536}

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class XE9680L_H200_SXM_141GBX8_HighAccuracy(XE9680L_H200_SXM_141GBX8):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE9680_H200_SXM_141GBX8(OfflineGPUBaseConfig):
    system = KnownSystem.XE9680_H200_SXM_141GBx8
    gpu_batch_size = {'llama2-70b': 2048}
    offline_expected_qps = 14.4 * 8 
    precision = "fp8"
    vboost_slider = 1
    trtllm_checkpoint_flags = {
        'kv_cache_dtype': 'fp8'
    }
    trtllm_build_flags = {
        'tensor_parallelism': 1,
        'pipeline_parallelism': 1,
        'max_num_tokens': 1536,
        'gemm_swiglu_plugin': 'fp8',
    }
    trtllm_runtime_flags = {'max_num_tokens': 1536}

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
class XE7745_H200_NVL_141GBx8(OfflineGPUBaseConfig):
    system = KnownSystem.XE7745_H200_NVL_141GBx8
    gpu_batch_size: dict = {'llama2-70b': 2048}
    offline_expected_qps = 160
    trtllm_build_flags = {
        'max_num_tokens': 1536,
        'gemm_swiglu_plugin': 'fp8',
    }
    trtllm_runtime_flags = {'max_num_tokens': 1536}

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class XE7745_H200_NVL_141GBx8_HighAccuracy(XE7745_H200_NVL_141GBx8):
    pass

