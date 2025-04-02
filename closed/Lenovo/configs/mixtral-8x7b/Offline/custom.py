# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.json

from . import *


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class SR650a_v4_H100_NVL_94GBx1(HopperOfflineGPUBaseConfig):
    system = KnownSystem.H100_NVL_94GBx1
    gpu_batch_size = {'mixtral-8x7b': 128*24}  #896
    offline_expected_qps = 36 #50 #46
    #trtllm_runtime_flags = {'max_num_tokens': 1024*8 } 


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class SR650a_v4_H100_NVL_94GBX4(SR650a_v4_H100_NVL_94GBx1):
    system = KnownSystem.H100_NVL_94GBx4
    offline_expected_qps = 36*4  #50*4 #46
    #numa_config = "0-1:0-85&2-3:86-171" #86

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class SR675V3_H200_SXMX4(HopperOfflineGPUBaseConfig):
    system = KnownSystem.SR675v3_H200_SXMx4

    precision = 'fp8'
    gpu_batch_size = {'mixtral-8x7b': 3072}
    trtllm_runtime_flags = {
        'max_num_tokens': 9 * 1024,
    }
    enable_sort = False
    tensor_parallelism = 1
    vboost_slider = 1

    offline_expected_qps = 60 * 4 * 0.97 #56

