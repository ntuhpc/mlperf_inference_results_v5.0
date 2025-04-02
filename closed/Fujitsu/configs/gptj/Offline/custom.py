
from . import *

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class CDI_H100NVLX8(HopperOfflineGPUBaseConfig):
    system = KnownSystem.H100_NVL_94GBx8
    gpu_batch_size = {'gptj': 192}
    #offline_expected_qps = 128
    offline_expected_qps = 250

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class CDI_H100NVLX8_HighAccuracy(CDI_H100NVLX8):
    #precision = "fp16"
    pass
