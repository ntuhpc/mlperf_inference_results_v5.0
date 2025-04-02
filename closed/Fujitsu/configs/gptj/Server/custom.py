
from . import *

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class CDI_H100NVLX8(HopperServerGPUBaseConfig):
    system = KnownSystem.H100_NVL_94GBx8
    gpu_batch_size = {'gptj': 128}
    #server_target_qps = 190
    server_target_qps = 215 #OK
    #server_target_qps =  240
    #server_target_qps = 110
    #server_target_qps = 120.5

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class CDI_H100NVLX8_HighAccuracy(CDI_H100NVLX8):
    pass
