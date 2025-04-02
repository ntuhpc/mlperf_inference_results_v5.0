# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.json

from . import *


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class C245M8_H100NVL_94GBX2(ServerGPUBaseConfig):
    system = KnownSystem.C245M8_H100NVL_94GBx2
    start_from_device = True
    gpu_copy_streams = 4
    use_deque_limit = True
    deque_timeout_usec = 30000
    gpu_batch_size = {'retinanet': 8}
    #gpu_batch_size = {'retinanet': 16}
    gpu_inference_streams = 2
    server_target_qps = 1200 * 2
    workspace_size = 60000000000

@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class X215M8_H100NVLx2(ServerGPUBaseConfig):
    system = KnownSystem.X215M8_H100NVLx2
    start_from_device = True
    gpu_copy_streams = 4
    use_deque_limit = True
    deque_timeout_usec = 30000
    gpu_batch_size = {'retinanet': 8}
    #gpu_batch_size = {'retinanet': 16}
    gpu_inference_streams = 2
    server_target_qps = 1200 * 2
    workspace_size = 60000000000

