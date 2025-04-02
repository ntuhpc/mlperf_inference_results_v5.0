# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.json

from . import *


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class SR675V3_H200_SXMX4(ServerGPUBaseConfig):
    system = KnownSystem.SR675v3_H200_SXMx4

    gpu_batch_size = {'clip1': 8 * 2, 'clip2': 8 * 2, 'unet': 8 * 2, 'vae': 8}
    use_graphs = False
    vboost_slider = 1

    server_target_qps = 2.1 * 4
    sdxl_batcher_time_limit = 4

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class SR650a_v4_H100_NVL_94GBx1(ServerGPUBaseConfig):
    system = KnownSystem.H100_NVL_94GBx1
    gpu_batch_size = {'clip1': 8 * 2, 'clip2': 8 * 2, 'unet': 8 * 2, 'vae': 8}
    server_target_qps = 2
    sdxl_batcher_time_limit = 3
    use_graphs = False  
    vboost_slider = 1


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class SR650a_v4_H100_NVL_94GBx4(SR650a_v4_H100_NVL_94GBx1):
    system = KnownSystem.H100_NVL_94GBx4
    server_target_qps = 6
    sdxl_batcher_time_limit = 5
