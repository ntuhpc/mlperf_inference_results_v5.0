from . import *


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE7745_L40SX8(ServerGPUBaseConfig):
    system = KnownSystem.XE7745_L40Sx8
    gpu_copy_streams = 4
    use_deque_limit = True
    deque_timeout_usec = 36911
    gpu_batch_size = {'retinanet': 8}
    gpu_inference_streams = 1
    server_target_qps = 6100
    workspace_size = 70000000000


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE9680L_H200_SXM_141GBX8(ServerGPUBaseConfig):
    system = KnownSystem.XE9680L_H200_SXM_141GBx8
    start_from_device = True
    gpu_copy_streams = 4
    use_deque_limit = True
    deque_timeout_usec = 30000
    gpu_batch_size = {'retinanet': 16}
    gpu_inference_streams = 2
    server_target_qps = 1700 * 8
    workspace_size = 60000000000


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE9680_H200_SXM_141GBX8(ServerGPUBaseConfig):
    system = KnownSystem.XE9680_H200_SXM_141GBx8
    start_from_device = True
    gpu_copy_streams = 4
    use_deque_limit = True
    deque_timeout_usec = 30000
    gpu_batch_size = {'retinanet': 16}
    gpu_inference_streams = 2
    server_target_qps = 1700 * 8
    workspace_size = 60000000000

