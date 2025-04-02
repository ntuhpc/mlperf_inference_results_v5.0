from . import *

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE9680L_H200_SXM_141GBX8(HopperServerGPUBaseConfig):
    system = KnownSystem.XE9680L_H200_SXM_141GBx8
    trtllm_build_flags = {'max_num_tokens': 8192}
    trtllm_runtime_flags = {'max_num_tokens': 8192}
    gpu_batch_size = {'mixtral-8x7b': 3072}
    server_target_qps = 50.5 * 8
