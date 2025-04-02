# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.json

from . import *


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class AS_4125GS_TNHR2_LCC_H200_SXM_141GBX8(H200_SXM_141GBx8):
    system = KnownSystem.AS_4125GS_TNHR2_LCC_H200_SXM_141GBX8

    server_target_qps = 14600


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class SYS_821GE_TNHR_H200_SXM_141GBX8(H200_SXM_141GBx8):
    system = KnownSystem.SYS_821GE_TNHR_H200_SXM_141GBX8

    server_target_qps = 14500


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class H100X8(ServerGPUBaseConfig):
    system = KnownSystem.h100x8
    start_from_device = True
    gpu_copy_streams = 4
    use_deque_limit = True
    deque_timeout_usec = 30000
    # gpu_batch_size = {'retinanet': 8}
    gpu_batch_size = {'retinanet': 8}
    gpu_inference_streams = 2
    server_target_qps = 1610 * 8
    workspace_size = 60000000000


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class SYS522GA_H200X8_NVL(ServerGPUBaseConfig):
    system = KnownSystem.SYS522GA_H200X8_NVL

    start_from_device = True
    gpu_copy_streams = 4
    use_deque_limit = True
    deque_timeout_usec = 30000
    gpu_batch_size = {'retinanet': 16}
    gpu_inference_streams = 2
    server_target_qps = 1410*8
    workspace_size = 60000000000

@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class SYS522GA_H200X8_NVL_Triton(SYS522GA_H200X8_NVL):
    use_triton = True

    server_target_qps = 1020 * 8
