from . import *

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE9680L_H200_SXM_141GBX8(HopperServerGPUBaseConfig):
    system = KnownSystem.XE9680L_H200_SXM_141GBx8
    gpu_batch_size = {'llama3.1-405b': 512}
    trtllm_build_flags = {
        'max_num_tokens': 8192,
        'tensor_parallelism': 8,
        'pipeline_parallelism': 1,
        # Disable to prevent intermittent failures;
        'gemm_allreduce_plugin': 'float16',
    }
    trtllm_runtime_flags = {
        'max_num_tokens': 2560,
        'max_batch_size': 64,
        'kvcache_free_gpu_mem_frac': 0.9
    }
    server_target_qps = 0.45


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE9680_H100_SXM_80GBX8(HopperServerGPUBaseConfig):
    system = KnownSystem.XE9680_H100_SXM_80GBx8
    gpu_batch_size = {'llama3.1-405b': 128}
    trtllm_build_flags = {
        'max_num_tokens': 8192,
        'tensor_parallelism': 4,
        'pipeline_parallelism': 2,
    }
    trtllm_runtime_flags = {
        'max_num_tokens': 2560,
        'max_batch_size': 64,
        'kvcache_free_gpu_mem_frac': 0.9,
    }
    server_target_qps = 0.41

