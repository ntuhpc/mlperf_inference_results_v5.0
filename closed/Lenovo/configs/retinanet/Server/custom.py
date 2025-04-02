# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.json

from . import *


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class SR675V3_H200_SXMX4(ServerGPUBaseConfig):
    system = KnownSystem.SR675v3_H200_SXMx4
    start_from_device = True
    gpu_copy_streams = 4
    use_deque_limit = True
    deque_timeout_usec = 30000
    gpu_inference_streams = 2
    workspace_size = 60000000000

    gpu_batch_size = {'retinanet': 14} #16->8 ==> (16+8)/2=12
    server_target_qps = 1700 * 4

@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class SR650a_v4_H100_NVL_94GBx4(ServerGPUBaseConfig):
    system = KnownSystem.H100_NVL_94GBx4
    start_from_device = True
    gpu_copy_streams = 4
    use_deque_limit = True
    deque_timeout_usec = 30000
    gpu_batch_size = {'retinanet': 16}
    gpu_inference_streams = 2
    server_target_qps = 1320*4 #1300
    workspace_size = 60000000000
    numa_config = "0-1:0-85&2-3:86-171"

