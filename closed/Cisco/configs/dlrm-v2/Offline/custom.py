# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.json

from . import *


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class C245M8_H100NVL94GBX2(OfflineGPUBaseConfig):
    system = KnownSystem.C245M8_H100NVL94GBx2
    gpu_batch_size = {'dlrm-v2': 51200}
    embedding_weights_on_gpu_part: float = 1.0
    offline_expected_qps = 102400


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class X215M8_H100NVLx2(OfflineGPUBaseConfig):
    system = KnownSystem.X215M8_H100NVLx2
    gpu_batch_size = {'dlrm-v2': 51200}
    embedding_weights_on_gpu_part: float = 1.0
    offline_expected_qps = 102500
