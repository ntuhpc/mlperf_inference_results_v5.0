# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.json

from . import *


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class C245M8_H100NVL_94GBx2(ServerGPUBaseConfig):
    system = KnownSystem.C245M8_H100NVL_94GBx2
    gpu_batch_size = {'dlrm-v2': 51200 * 2}
    embedding_weights_on_gpu_part: float = 1.0
    server_target_qps = 98600
    vboost_slider = 1  


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class X215M8_H100NVLx2(ServerGPUBaseConfig):
    system = KnownSystem.X215M8_H100NVLx2
    gpu_batch_size = {'dlrm-v2': 51200 * 2}
    embedding_weights_on_gpu_part: float = 1.0
    server_target_qps = 98600
    vboost_slider = 1
