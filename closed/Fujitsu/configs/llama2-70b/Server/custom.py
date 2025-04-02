
from . import *

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP, "TP2")
class CDI_H100NVLX8(HopperServerGPUBaseConfig):
    system = KnownSystem.H100_NVL_94GBx8

    gpu_batch_size = {'llama2-70b': 640}
    server_target_qps = 58.5
    trtllm_build_flags = {
        'tensor_parallelism': 2,
        'pipeline_parallelism': 1,
    }

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP, "TP2")
class CDI_H100NVLX8_HighAccuracy(CDI_H100NVLX8):
    pass
