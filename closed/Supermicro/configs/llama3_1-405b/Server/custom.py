# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.json

from . import *


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class SYS_A21GE_NBRT_B200_SXM_180GBX8(B200_SXM_180GB_TP2PP2x2):
    system = KnownSystem.SYS_A21GE_NBRT_B200_SXM_180GBx8

    server_target_qps = 0.83 * 2


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class SYS_421GE_NBRT_LCC_B200_SXM_180GBX8(B200_SXM_180GB_TP2PP2x2):
    system = KnownSystem.SYS_421GE_NBRT_LCC_B200_SXM_180GBx8

    server_target_qps = 0.81 * 2


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class AS_4125GS_TNHR2_LCC_H200_SXM_141GBX8(H200_SXM_141GBx8_TP8PP1):
    system = KnownSystem.AS_4125GS_TNHR2_LCC_H200_SXM_141GBX8

