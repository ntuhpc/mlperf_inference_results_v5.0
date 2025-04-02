from . import *


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE7745_L40SX8(OfflineGPUBaseConfig):
    system = KnownSystem.XE7745_L40Sx8
    gpu_batch_size = {'retinanet': 4}
    gpu_copy_streams = 2
    gpu_inference_streams = 2
    offline_expected_qps = 7000
    run_infer_on_copy_streams = False
    workspace_size = 60000000000


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE9680L_H200_SXM_141GBX8(OfflineGPUBaseConfig):
    system = KnownSystem.XE9680L_H200_SXM_141GBx8
    gpu_batch_size = {'retinanet': 48}
    gpu_copy_streams = 2
    gpu_inference_streams = 2
    offline_expected_qps = 1800 * 8
    run_infer_on_copy_streams = True
    workspace_size = 200000000000


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE9680_H200_SXM_141GBX8(OfflineGPUBaseConfig):
    system = KnownSystem.XE9680_H200_SXM_141GBx8
    gpu_batch_size = {'retinanet': 48}
    gpu_copy_streams = 2
    gpu_inference_streams = 2
    offline_expected_qps = 1700 * 8
    run_infer_on_copy_streams = False
    workspace_size = 128000000000

