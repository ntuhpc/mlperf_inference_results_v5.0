# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.json

from . import *


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class C245M8_H100NVL_94GBX2(HopperServerGPUBaseConfig):
    system = KnownSystem.C245M8_H100NVL_94GBx2
    gpu_batch_size: dict = {'gptj': 128}
    use_fp8 = True
    server_target_qps = 48

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class C245M8_H100NVL_94GBX2_HighAccuracy(C245M8_H100NVL_94GBX2):
    pass


