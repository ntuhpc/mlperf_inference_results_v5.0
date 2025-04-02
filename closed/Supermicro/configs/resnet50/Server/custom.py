# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.json

from . import *


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class SYS_821GE_TNHR_H200_SXM_141GBX8(H200_SXM_141GBx8):
    system = KnownSystem.SYS_821GE_TNHR_H200_SXM_141GBX8

    server_target_qps = 640000

    
@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class H100X8(ServerGPUBaseConfig):
    system = KnownSystem.h100x8
    use_deque_limit = True
    deque_timeout_usec = 3000
    gpu_batch_size = {'resnet50': 256}
    gpu_copy_streams = 4
    gpu_inference_streams = 7
    server_target_qps = 75000 * 8
    use_cuda_thread_per_device = True
    use_graphs = True
    start_from_device = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class AS8125GSTNHR(ServerGPUBaseConfig):
    system = KnownSystem.AS8125GSTNHR
    use_deque_limit = True
    deque_timeout_usec = 3000
    gpu_batch_size = {'resnet50': 256}
    gpu_copy_streams = 4
    gpu_inference_streams = 7
    server_target_qps = 73000 * 8
    use_cuda_thread_per_device = True
    use_graphs = True
    start_from_device = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class SYS522GA_H200X8_NVL(ServerGPUBaseConfig):
    system = KnownSystem.SYS522GA_H200X8_NVL

    use_deque_limit = True
    deque_timeout_usec = 3000
    gpu_batch_size = {'resnet50': 224}
    server_target_qps = 65000*8
    use_cuda_thread_per_device = True
    use_graphs = True
    start_from_device = True
    gpu_copy_streams = 4
    gpu_inference_streams = 7


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class SYS522GA_H200X8_NVL_Triton(SYS522GA_H200X8_NVL):
    use_triton = True

    deque_timeout_usec = 500
    gpu_batch_size = {'resnet50': 64}
    gpu_inference_streams = 5
    server_target_qps = 55000*8
    use_graphs = False
    use_triton = True
    batch_triton_requests = True
    max_queue_delay_usec = 1000
    request_timeout_usec = 2000
