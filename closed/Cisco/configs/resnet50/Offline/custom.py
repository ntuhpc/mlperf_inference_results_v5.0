# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.json

from . import *


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class C245M8_H100NVL_94GBX2(OfflineGPUBaseConfig):
    system = KnownSystem.C245M8_H100NVL_94GBx2
    gpu_batch_size = {'resnet50': 2048}
    offline_expected_qps = 142000    

@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class X215M8_H100NVLx2(OfflineGPUBaseConfig):
    system = KnownSystem.X215M8_H100NVLx2
    gpu_batch_size = {'resnet50': 2048}
    offline_expected_qps = 142000

