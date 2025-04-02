# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.json

from . import *


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class C245M8_H100NVL_94GBX2(ServerGPUBaseConfig):
    system = KnownSystem.C245M8_H100NVL_94GBx2
    gpu_batch_size = {'clip1': 8 * 2, 'clip2': 8 * 2, 'unet': 8 * 2, 'vae': 8}
    server_target_qps = 3
    sdxl_batcher_time_limit = 3
    use_graphs = False  # disable to meet latency constraint for x1
    vboost_slider = 1
