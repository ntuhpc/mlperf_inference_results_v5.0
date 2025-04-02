# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *

''' 
@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class HPE_ProLiant_DL380a_L40S_PCIe_48GBx4(ServerGPUBaseConfig):

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class HPE_ProLiant_DL380a_L40S_PCIe_48GBx4_HighAccuracy(HPE_ProLiant_DL380a_L40S_PCIe_48GBx4):
    pass

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class HPE_ProLiant_DL380a_H100_PCIe_80GBx4(ServerGPUBaseConfig):
    system = KnownSystem.HPE_ProLiant_DL380a_H100_PCIe_80GBx4

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class HPE_ProLiant_DL380a_H100_PCIe_80GBx4_HighAccuracy(HPE_ProLiant_DL380a_H100_PCIe_80GBx4):
    pass

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class HPE_PROLIANT_DL380A_H100_NVL_94GBX4(H100_NVL_94GB_TP2x2):
    system = KnownSystem.HPE_ProLiant_DL380a_H100_NVL_94GBx4

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class HPE_PROLIANT_DL380A_H100_NVL_94GBX4_HighAccuracy(HPE_PROLIANT_DL380A_H100_NVL_94GBX4):
    pass
 '''
@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class HPE_CRAY_XD670_H100_SXM_80GBX8(H200_SXM_141GBx8):
    system = KnownSystem.HPE_Cray_XD670_H100_SXM_80GBx8
    server_target_qps = 7.84*8
    trtllm_build_flags = {
        'tensor_parallelism': 2,
        'pipeline_parallelism': 1,
    }

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class HPE_CRAY_XD670_H100_SXM_80GBX8_HighAccuracy(HPE_CRAY_XD670_H100_SXM_80GBX8):
    pass

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class HPE_CRAY_XD670_H200_SXM_141GBX8(H200_SXM_141GBx8):
    system = KnownSystem.HPE_Cray_XD670_H200_SXM_141GBx8
    precision = 'fp8'
    vboost_slider = 1
    trtllm_build_flags = {
        'tensor_parallelism': 1,
        'pipeline_parallelism': 1,
    }
    trtllm_checkpoint_flags = {
        'kv_cache_dtype': 'fp8'
    }
    server_target_qps = 8.4*8

    # #ServerGPUBaseConfig(GPUBaseConfig):
    # scenario = Scenario.Server
    # min_duration = 1200000
    # enable_sort = False
    # #HopperServerGPUBaseConfig(ServerGPUBaseConfig):
    # precision = 'fp8'
    # vboost_slider = 1
    # trtllm_build_flags = {
    #     'tensor_parallelism': 1,
    #     'pipeline_parallelism': 1,
    # }
    # trtllm_checkpoint_flags = {
    #     'kv_cache_dtype': 'fp8'
    # }
    # trtllm_runtime_flags = {
    #     'kvcache_free_gpu_mem_frac': 0.90
    # }
    # #H200_SXM_141GBx1(HopperServerGPUBaseConfig):
    # gpu_batch_size = {'llama2-70b-interactive': 512}
    # trtllm_build_flags = {'max_num_tokens': 256}
    # trtllm_runtime_flags = {'max_num_tokens': 256}
    # server_target_qps = 8.4
    # #H200_SXM_141GBx8(H200_SXM_141GBx1):
    # server_target_qps = H200_SXM_141GBx1.server_target_qps * 8

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class HPE_CRAY_XD670_H200_SXM_141GBX8_HighAccuracy(HPE_CRAY_XD670_H200_SXM_141GBX8):
    pass



