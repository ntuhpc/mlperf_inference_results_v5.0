from . import *

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE9680L_H200_SXM_141GBX8(HopperOfflineGPUBaseConfig):
    system = KnownSystem.XE9680L_H200_SXM_141GBx8
    gpu_batch_size = {'mixtral-8x7b': 3072}
    offline_expected_qps = 56 * 8 * 0.97
    trtllm_runtime_flags = {
        'max_num_tokens': 16 * 1024,
    }

