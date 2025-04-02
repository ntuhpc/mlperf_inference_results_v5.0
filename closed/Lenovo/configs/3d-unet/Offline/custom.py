# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.json

from . import *


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class SR675V3_H200_SXMX4(OfflineGPUBaseConfig):
    system = KnownSystem.SR675v3_H200_SXMx4
    workspace_size = 128000000000
    gpu_batch_size = {'3d-unet': 8}
    offline_expected_qps = 6.8 * 4


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class SR675V3_H200_SXMX4_HighAccuracy(SR675V3_H200_SXMX4):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class SR650a_v4_H100_NVL_94GBx4(OfflineGPUBaseConfig):
    system = KnownSystem.H100_NVL_94GBx4
    gpu_batch_size = {'3d-unet': 8}
    offline_expected_qps = 24 #22.4 #17.5
    numa_config = "0-1:0-85&2-3:86-171"

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class SR650a_v4_H100_NVL_94GBx4_HighAccuracy(SR650a_v4_H100_NVL_94GBx4):
    pass
