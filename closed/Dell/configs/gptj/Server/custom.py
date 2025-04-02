from . import *

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE7745_L40SX8(L40Sx8):
    system = KnownSystem.XE7745_L40Sx8
    gpu_batch_size = {'gptj': 112}
    precision = 'fp8'
    server_target_qps = 80
    enable_sort= False
    trtllm_checkpoint_flags = {
        'kv_cache_dtype': 'fp8'
    }

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class XE7745_L40SX8_HighAccuracy(XE7745_L40SX8):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE9680L_H200_SXM_141GBX8(HopperServerGPUBaseConfig):
    system = KnownSystem.XE9680L_H200_SXM_141GBx8
    gpu_batch_size = {'gptj': 396}
    server_target_qps = 39 * 8

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class XE9680L_H200_SXM_141GBX8_HighAccuracy(XE9680L_H200_SXM_141GBX8):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE9680_H200_SXM_141GBX8(HopperServerGPUBaseConfig):
    system = KnownSystem.XE9680_H200_SXM_141GBx8
    gpu_batch_size = {'gptj': 396}
    server_target_qps = 39 * 8

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class XE9680_H200_SXM_141GBX8_HighAccuracy(XE9680_H200_SXM_141GBX8):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE9680_H100_SXM_80GBX8(HopperServerGPUBaseConfig):
    system = KnownSystem.XE9680_H100_SXM_80GBx8
    gpu_batch_size = {'gptj': 256}
    server_target_qps = 34.92 * 8 

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class XE9680_H100_SXM_80GBX8_HighAccuracy(XE9680_H100_SXM_80GBX8):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE7745_H200_NVL_141GBx8(HopperServerGPUBaseConfig):
    system = KnownSystem.XE7745_H200_NVL_141GBx8
    gpu_batch_size = {'gptj': 512}
    server_target_qps = 260

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class XE7745_H200_NVL_141GBx8_HighAccuracy(XE7745_H200_NVL_141GBx8):
    pass

