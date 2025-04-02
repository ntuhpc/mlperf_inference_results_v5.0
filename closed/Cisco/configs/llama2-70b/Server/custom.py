# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.json

from . import *


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class C245M8_H100NVL_94GBX2(HopperServerGPUBaseConfig):
    system = KnownSystem.C245M8_H100NVL_94GBx2
    gpu_batch_size = {'llama2-70b': 640}
    server_target_qps = 11.5
    trtllm_build_flags = {
        'tensor_parallelism': 1,
        'pipeline_parallelism': 1,
    }


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class C245M8_H100NVL_94GBX2_HighAccuracy(C245M8_H100NVL_94GBX2):
    pass


