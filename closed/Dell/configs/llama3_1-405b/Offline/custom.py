from . import *

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE9680L_H200_SXM_141GBX8(HopperOfflineGPUBaseConfig):
    system = KnownSystem.XE9680L_H200_SXM_141GBx8
    offline_expected_qps = 0.9
    trtllm_build_flags = {
        'max_num_tokens': 2560,
        'tensor_parallelism': 4,
        'pipeline_parallelism': 2,
        # Disable to prevent intermittent failures
        'gemm_allreduce_plugin': 'float16',
    }
    trtllm_runtime_flags = {'max_num_tokens': 1536}
    gpu_batch_size = {'llama3.1-405b': 512}


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE9680_H100_SXM_80GBx8(HopperOfflineGPUBaseConfig):
    system = KnownSystem.XE9680_H100_SXM_80GBx8
    offline_expected_qps = 0.9
    trtllm_build_flags = {
        'max_num_tokens': 2560,
        'tensor_parallelism': 8,
        'pipeline_parallelism': 1,
    }
    trtllm_runtime_flags = {'max_num_tokens': 1536}
    gpu_batch_size = {'llama3.1-405b': 512}

