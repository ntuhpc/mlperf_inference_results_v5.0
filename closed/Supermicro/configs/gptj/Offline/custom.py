# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.json

from . import *


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class AS_4125GS_TNHR2_LCC_H200_SXM_141GBX8(H200_SXM_141GBx8):
    system = KnownSystem.AS_4125GS_TNHR2_LCC_H200_SXM_141GBX8


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class AS_4125GS_TNHR2_LCC_H200_SXM_141GBX8_HighAccuracy(H200_SXM_141GBx8):
    system = KnownSystem.AS_4125GS_TNHR2_LCC_H200_SXM_141GBX8


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class SYS_821GE_TNHR_H200_SXM_141GBX8(H200_SXM_141GBx8):
    system = KnownSystem.SYS_821GE_TNHR_H200_SXM_141GBX8


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class SYS_821GE_TNHR_H200_SXM_141GBX8_HighAccuracy(H200_SXM_141GBx8):
    system = KnownSystem.SYS_821GE_TNHR_H200_SXM_141GBX8


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class H100X8(OfflineGPUBaseConfig):
    system = KnownSystem.h100x8
    gpu_batch_size = {'gptj': 192}
    precision = "fp8"
    offline_expected_qps = 36 * 8


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class H100X8_HighAccuracy(H100X8):
    offline_expected_qps = 36 * 8


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class SYS522GA_H200X8_NVL(OfflineGPUBaseConfig):
    system = KnownSystem.SYS522GA_H200X8_NVL


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class SYS522GA_H200X8_NVL_HighAccuracy(SYS522GA_H200X8_NVL):
    pass