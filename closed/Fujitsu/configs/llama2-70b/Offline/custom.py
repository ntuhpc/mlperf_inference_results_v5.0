
from . import *
@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP, "TP2")
class CDI_H100NVLX8(HopperOfflineGPUBaseConfig):
    system = KnownSystem.H100_NVL_94GBx8
    gpu_batch_size = {'llama2-70b': 1300}
    offline_expected_qps = 65
    trtllm_build_flags = {
        'tensor_parallelism': 2,
        'pipeline_parallelism': 1,
    }

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP, "TP2")
class CDI_H100NVLX8_HighAccuracy(CDI_H100NVLX8):
    pass

"""
@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP, "PP2")
class CDI_H100NVLX8(HopperOfflineGPUBaseConfig):
    system = KnownSystem.H100_NVL_94GBx8
    gpu_batch_size = {'llama2-70b': 1300}
    offline_expected_qps = 65
    trtllm_build_flags = {
        'tensor_parallelism': 1,
        'pipeline_parallelism': 2,
    }

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP, "PP2")
class CDI_H100NVLX8_HighAccuracy(CDI_H100NVLX8):
    pass
"""
