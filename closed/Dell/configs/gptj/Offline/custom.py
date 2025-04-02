from . import *

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE7745_L40SX8(OfflineGPUBaseConfig):
    system = KnownSystem.XE7745_L40Sx8
    gpu_batch_size = {'gptj': 100}
    precision = 'fp8'
    offline_expected_qps = 110.7
    vboost_slider = 1
    enable_sort= False
    trtllm_checkpoint_flags = {
        'kv_cache_dtype': 'fp8'
    }

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class XE7745_L40SX8_HighAccuracy(XE7745_L40SX8):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE9680L_H200_SXM_141GBX8(HopperOfflineGPUBaseConfig):
    system = KnownSystem.XE9680L_H200_SXM_141GBx8
    gpu_batch_size = {'gptj': 396}
    offline_expected_qps = 36 * 8
 
@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class XE9680L_H200_SXM_141GBX8_HighAccuracy(XE9680L_H200_SXM_141GBX8):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE9680_H200_SXM_141GBX8(HopperOfflineGPUBaseConfig):
    system = KnownSystem.XE9680_H200_SXM_141GBx8
    gpu_batch_size = {'gptj': 396}
    offline_expected_qps = 36 * 8

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class XE9680_H200_SXM_141GBX8_HighAccuracy(XE9680_H200_SXM_141GBX8):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE9680_H100_SXM_80GBX8(HopperOfflineGPUBaseConfig):
    system = KnownSystem.XE9680_H100_SXM_80GBx8
    gpu_batch_size = {'gptj': 256}
    offline_expected_qps = 40 * 8

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class XE9680_H100_SXM_80GBX8_HighAccuracy(XE9680_H100_SXM_80GBX8):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE7745_H200_NVL_141GBx8(HopperOfflineGPUBaseConfig):
    system = KnownSystem.XE7745_H200_NVL_141GBx8
    gpu_batch_size = {'gptj': 512}
    offline_expected_qps = 500

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class XE7745_H200_NVL_141GBx8_HighAccuracy(XE7745_H200_NVL_141GBx8):
    pass

