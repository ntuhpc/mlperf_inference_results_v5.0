# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class SYS_A21GE_NBRT_B200_SXM_180GBX8(B200_SXM_180GBx8):
    system = KnownSystem.SYS_A21GE_NBRT_B200_SXM_180GBx8


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class SYS_A21GE_NBRT_B200_SXM_180GBX8_HighAccuracy(B200_SXM_180GBx8_HighAccuracy):
    system = KnownSystem.SYS_A21GE_NBRT_B200_SXM_180GBx8


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class SYS_421GE_NBRT_LCC_B200_SXM_180GBX8(B200_SXM_180GBx8):
    system = KnownSystem.SYS_421GE_NBRT_LCC_B200_SXM_180GBx8


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class SYS_421GE_NBRT_LCC_B200_SXM_180GBX8_HighAccuracy(B200_SXM_180GBx8_HighAccuracy):
    system = KnownSystem.SYS_421GE_NBRT_LCC_B200_SXM_180GBx8


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
class ARS_111GL_NHR(HopperOfflineGPUBaseConfig):
    system = KnownSystem.ARS_111GL_NHR

    gpu_batch_size = {'llama2-70b': 2048}
    offline_expected_qps = 15.0
    trtllm_build_flags = {
        'max_num_tokens': 1536,
        'gemm_swiglu_plugin': 'fp8',
    }
    trtllm_runtime_flags = {'max_num_tokens': 1536}


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class ARS_111GL_NHR_HighAccuracy(ARS_111GL_NHR):
    pass
