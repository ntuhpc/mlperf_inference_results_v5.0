# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.json

from . import *

@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class SR650a_v4_H100_NVL_94GBx4(H100_NVL_94GBx1):
    system = KnownSystem.H100_NVL_94GBx4
    gpu_batch_size = {'resnet50': 2674} #2650
    offline_expected_qps = 288000
    numa_config = "0-1:0-85&2-3:86-171"

@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class SR675V3_H200_SXMX4(OfflineGPUBaseConfig):
    system = KnownSystem.SR675v3_H200_SXMx4

    gpu_batch_size = {'resnet50': 2048}
    offline_expected_qps = 600000 #450000 #380000
    start_from_device = True
    numa_config = "0,1:0-95&2,3:96-191"

