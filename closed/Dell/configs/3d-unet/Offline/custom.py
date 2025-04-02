from . import *

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class R7725_H100NVL_PCIE_94GBX2(OfflineGPUBaseConfig):
    system = KnownSystem.R7725_H100NVL_PCIE_94GBx2
    gpu_batch_size = {'3d-unet': 4}
    offline_expected_qps = 24

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class R7725_H100NVL_PCIE_94GBX2_HighAccuracy(R7725_H100NVL_PCIE_94GBX2):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE7745_L40SX8(OfflineGPUBaseConfig):
    system = KnownSystem.XE7745_L40Sx8
    gpu_inference_streams = 1
    gpu_copy_streams = 1
    gpu_batch_size = {'3d-unet': 1}
    offline_expected_qps = 60
    slice_overlap_patch_kernel_cg_impl = True

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class XE7745_L40SX8_HighAccuracy(XE7745_L40SX8):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE9680L_H200_SXM_141GBX8(OfflineGPUBaseConfig):
    system = KnownSystem.XE9680L_H200_SXM_141GBx8
    workspace_size = 128000000000
    gpu_batch_size = {'3d-unet': 8}
    offline_expected_qps = 6.8 * 8

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class XE9680L_H200_SXM_141GBX8_HighAccuracy(XE9680L_H200_SXM_141GBX8):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE9680_H200_SXM_141GBX8(OfflineGPUBaseConfig):
    system = KnownSystem.XE9680_H200_SXM_141GBx8
    workspace_size = 128000000000
    gpu_batch_size = {'3d-unet': 8}
    offline_expected_qps = 6.8 * 8

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class XE9680_H200_SXM_141GBX8_HighAccuracy(XE9680_H200_SXM_141GBX8):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE9680_H100_SXM_80GBX8(OfflineGPUBaseConfig):
    system = KnownSystem.XE9680_H100_SXM_80GBx8
    gpu_batch_size = {'3d-unet': 8}
    offline_expected_qps = 6.8 * 8

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class XE9680_H100_SXM_80GBX8_HighAccuracy(XE9680_H100_SXM_80GBX8):
    pass
