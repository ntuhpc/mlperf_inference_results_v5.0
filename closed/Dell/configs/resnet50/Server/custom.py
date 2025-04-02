from . import *


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE7745_L40SX8(ServerGPUBaseConfig):
    system = KnownSystem.XE7745_L40Sx8
    use_deque_limit = True
    deque_timeout_usec = 2000
    gpu_batch_size = {'resnet50': 80}
    gpu_copy_streams = 9
    gpu_inference_streams = 2
    server_target_qps = 345000
    use_cuda_thread_per_device = True
    use_graphs = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE9680L_H200_SXM_141GBX8(ServerGPUBaseConfig):
    system = KnownSystem.XE9680L_H200_SXM_141GBx8
    use_deque_limit = True
    deque_timeout_usec = 3000
    gpu_batch_size = {'resnet50': 256}
    gpu_copy_streams = 4
    gpu_inference_streams = 7
    server_target_qps = 79000 * 8
    use_cuda_thread_per_device = True
    use_graphs = True
    start_from_device = True

