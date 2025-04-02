# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.json

from . import *


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class C245M8_H100NVL_94GBX2(ServerGPUBaseConfig):
    system = KnownSystem.C245M8_H100NVL_94GBx2
    use_deque_limit = True
    deque_timeout_usec = 3000
    gpu_batch_size = {'resnet50': 256}
    gpu_copy_streams = 2
    gpu_inference_streams = 1
    server_target_qps = 120000
    use_cuda_thread_per_device = True
    use_graphs = True
    start_from_device = True

@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class X215M8_H100NVLx2(ServerGPUBaseConfig):
    system = KnownSystem.X215M8_H100NVLx2
    use_deque_limit = True
    deque_timeout_usec = 3000
    gpu_batch_size = {'resnet50': 256}
    gpu_copy_streams = 2
    gpu_inference_streams = 1
    server_target_qps = 120000
    use_cuda_thread_per_device = True
    use_graphs = True
    start_from_device = True



