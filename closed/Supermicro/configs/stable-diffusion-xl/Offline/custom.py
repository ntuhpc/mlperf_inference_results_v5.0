# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.json

from . import *


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class SYS_A21GE_NBRT_B200_SXM_180GBX8(B200_SXM_180GBx8):
    system = KnownSystem.SYS_A21GE_NBRT_B200_SXM_180GBx8


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class SYS_421GE_NBRT_LCC_B200_SXM_180GBX8(B200_SXM_180GBx8):
    system = KnownSystem.SYS_421GE_NBRT_LCC_B200_SXM_180GBx8


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class AS_4125GS_TNHR2_LCC_H200_SXM_141GBX8(H200_SXM_141GBx8):
    system = KnownSystem.AS_4125GS_TNHR2_LCC_H200_SXM_141GBX8


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class SYS_821GE_TNHR_H200_SXM_141GBX8(H200_SXM_141GBx8):
    system = KnownSystem.SYS_821GE_TNHR_H200_SXM_141GBX8

    
@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class H100X8(OfflineGPUBaseConfig):
    system = KnownSystem.h100x8
    gpu_batch_size = {'clip1': 32 * 2, 'clip2': 32 * 2, 'unet': 32 * 2, 'vae': 8}
    offline_expected_qps = 17
    use_graphs = False
    vboost_slider = 1


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class SYS522GA_H200X8_NVL(OfflineGPUBaseConfig):
    system = KnownSystem.SYS522GA_H200X8_NVL
    gpu_batch_size = {'clip1': 32 * 2, 'clip2': 32 * 2, 'unet': 32 * 2, 'vae': 8}
    offline_expected_qps = 15
    use_graphs = False
    vboost_slider = 1