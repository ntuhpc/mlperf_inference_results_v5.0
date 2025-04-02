# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.json

from . import *

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class SR650a_v4_H100_NVL_94GBx1(HopperServerGPUBaseConfig):
    system = KnownSystem.H100_NVL_94GBx1
    gpu_batch_size = {'mixtral-8x7b': 128*24} #1536  #896
    server_target_qps = 33   #40  #43.5
    #trtllm_build_flags = {'max_num_tokens': 16384}
    #trtllm_runtime_flags = {'max_num_tokens': 16384} #8192
    #tensor_parallelism = 1


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class SR650a_v4_H100_NVL_94GBx4(SR650a_v4_H100_NVL_94GBx1):
    system = KnownSystem.H100_NVL_94GBx4
    server_target_qps = 33*4 # 40 * 4  #43.5
    #numa_config = "0-1:0-85&2-3:86-171" #86

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class SR675V3_H200_SXMX4(HopperServerGPUBaseConfig):
    system = KnownSystem.SR675v3_H200_SXMx4

    precision = 'fp8'
    trtllm_build_flags = {'max_num_tokens': 16384} #8192
    trtllm_runtime_flags = {'max_num_tokens': 16384} #8192
    gpu_batch_size = {'mixtral-8x7b': 1200}
    server_target_qps = 50 * 4 #55,52 X #49