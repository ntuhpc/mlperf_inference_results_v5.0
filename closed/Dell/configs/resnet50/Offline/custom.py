from . import *


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE7745_L40SX8(OfflineGPUBaseConfig):
    system = KnownSystem.XE7745_L40Sx8
    gpu_batch_size = {'resnet50': 64}
    gpu_inference_streams = 1
    gpu_copy_streams = 2
    offline_expected_qps = 400000


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE9680L_H200_SXM_141GBX8(OfflineGPUBaseConfig):
    system = KnownSystem.XE9680L_H200_SXM_141GBx8
    gpu_batch_size = {'resnet50': 2048}
    offline_expected_qps = 90000 * 8
    start_from_device = True

