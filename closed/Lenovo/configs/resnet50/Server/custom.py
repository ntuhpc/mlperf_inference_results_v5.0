# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.json

from . import *

@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class SR650a_v4_H100_NVL_94GBx4(ServerGPUBaseConfig):
    system = KnownSystem.H100_NVL_94GBx4
    use_deque_limit = True
    deque_timeout_usec = 4182 #3000
    gpu_batch_size = {'resnet50': 391} #384
    gpu_copy_streams = 2  #2
    gpu_inference_streams = 1
    server_target_qps = 66300*4 # 66000 
    use_cuda_thread_per_device = True
    use_batcher_thread_per_device = True
    use_graphs = True
    start_from_device = True
    numa_config = "0-1:0-85&2-3:86-171" #86


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class SR675V3_H200_SXMX4(ServerGPUBaseConfig):
    system = KnownSystem.SR675v3_H200_SXMx4

    use_deque_limit = True
    deque_timeout_usec = 3000 #4182 #3000
    gpu_batch_size = {'resnet50': 391} #256
    use_cuda_thread_per_device = True
    use_graphs = True
    start_from_device = True

    gpu_copy_streams = 3 #4
    gpu_inference_streams = 4 #7
    server_target_qps = 76000 * 4 #79000 * 4
    numa_config = "0,1:0-95&2,3:96-191"

