# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.json

from . import *


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class CDI_H100NVLX8(OfflineGPUBaseConfig):
    system = KnownSystem.H100_NVL_94GBx8
    gpu_batch_size = {'retinanet': 48}
    gpu_copy_streams = 2
    gpu_inference_streams = 2
    run_infer_on_copy_streams = False
    workspace_size = 60000000000
    offline_expected_qps = 12000




