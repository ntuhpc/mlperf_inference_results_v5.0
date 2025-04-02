# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.json

from . import *


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class SYS_821GE_TNHR_H200_SXM_141GBX8(H200_SXM_141GBx8):
    system = KnownSystem.SYS_821GE_TNHR_H200_SXM_141GBX8


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class H100X8(OfflineGPUBaseConfig):
    system = KnownSystem.h100x8
    gpu_batch_size = {'resnet50': 2048}
    offline_expected_qps = 90000 * 8
    start_from_device = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class SYS522GA_H200X8_NVL(OfflineGPUBaseConfig):
    system = KnownSystem.SYS522GA_H200X8_NVL


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class SYS522GA_H200X8_NVL_Triton(SYS522GA_H200X8_NVL):
    use_triton = True

