# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.json

from . import *


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class SR675V3_H200_SXMX4(OfflineGPUBaseConfig):
    system = KnownSystem.SR675v3_H200_SXMx4

    gpu_batch_size = {'retinanet': 48}
    gpu_copy_streams = 2
    gpu_inference_streams = 2
    offline_expected_qps = 1850 * 4
    run_infer_on_copy_streams = False
    workspace_size = 128000000000
    
@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class SR650a_v4_H100_NVL_94GBx4(OfflineGPUBaseConfig):
    system = KnownSystem.H100_NVL_94GBx4
    gpu_batch_size = {'retinanet': 60} #48
    gpu_copy_streams = 2
    gpu_inference_streams = 2
    offline_expected_qps = 1880 * 4 # 1100 
    run_infer_on_copy_streams = False
    workspace_size = 60000000000
    numa_config = "0-1:0-85&2-3:86-171" #86

