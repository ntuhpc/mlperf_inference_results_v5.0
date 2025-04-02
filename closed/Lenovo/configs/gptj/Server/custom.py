# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.json

from . import *

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class SR650a_v4_H100_NVL_94GBx1(HopperServerGPUBaseConfig):
    system = KnownSystem.H100_NVL_94GBx1
    gpu_batch_size = {'gptj': 256}
    server_target_qps = 24.5 # 20 # 16.2
    gpu_copy_streams = 2 #1																														
    gpu_inference_streams = 1																														
    tensor_parallelism = 1																														
    #precision = "fp16"																														
    #enable_sort = False																														
    num_sort_segments = 2																														
    #use_token_latencies = False																														


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class SR650a_v4_H100_NVL_94GBx4(SR650a_v4_H100_NVL_94GBx1):
    system = KnownSystem.H100_NVL_94GBx4
    use_fp8 = True
    gpu_batch_size = {'gptj': 256}
    server_target_qps = 24.5 * 4 #20


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class SR650a_v4_H100_NVL_94GBx1_HighAccuracy(SR650a_v4_H100_NVL_94GBx1):
     pass #precision = "fp16"


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class SR650a_v4_H100_NVL_940GBx4_HighAccuracy(SR650a_v4_H100_NVL_94GBx4):
     pass #precision = "fp16"


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class SR675V3_H200_SXMX4(HopperServerGPUBaseConfig):
    system = KnownSystem.SR675v3_H200_SXMx4
    gpu_batch_size = {'gptj': 396}
    server_target_qps = 38 * 4 #34.92


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class SR675V3_H200_SXMX4_HighAccuracy(SR675V3_H200_SXMX4):
    pass


