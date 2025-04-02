# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.json

from . import *


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class CDI_H100NVLX8(OfflineGPUBaseConfig):
    system = KnownSystem.H100_NVL_94GBx8
    #gpu_batch_size = {'rgat': 4092}
    #offline_expected_qps = 300_000

    #gpu_batch_size = {'rgat': 3092}
    #offline_expected_qps = 200_000

    gpu_batch_size = {'rgat': 768}
    #offline_expected_qps = 56250
    #offline_expected_qps = 70000
    offline_expected_qps = 77000


