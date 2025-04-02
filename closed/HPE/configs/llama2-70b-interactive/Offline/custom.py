# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *

''' 
@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class HPE_ProLiant_DL380a_L40S_PCIe_48GBx4(OfflineGPUBaseConfig):
    system = KnownSystem.HPE_ProLiant_DL380a_L40S_PCIe_48GBx4

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class HPE_ProLiant_DL380a_L40S_PCIe_48GBx4_HighAccuracy(HPE_ProLiant_DL380a_L40S_PCIe_48GBx4):
    pass

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class HPE_ProLiant_DL380a_H100_PCIe_80GBx4(OfflineGPUBaseConfig):
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
    offline_expected_qps = 14.1*8
    trtllm_build_flags = {
        'use_paged_context_fmha': 'enable',
        'tokens_per_block': 32,
        'tensor_parallelism': 2,
        'pipeline_parallelism': 1,
    }

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class HPE_CRAY_XD670_H100_SXM_80GBX8_HighAccuracy(HPE_CRAY_XD670_H100_SXM_80GBX8):
    pass

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class HPE_CRAY_XD670_H200_SXM_141GBX8(H200_SXM_141GBx8):
    system = KnownSystem.HPE_Cray_XD670_H200_SXM_141GBx8
    offline_expected_qps = 14.4*8

    # #OfflineGPUBaseConfig(GPUBaseConfig):
    # min_duration = 2400000
    # enable_sort = False
    # trtllm_runtime_flags = {
    #     'batch_scheduler_policy': 'max_util',
    #     'context_chunking_policy': 'first_come_first_served',
    # }
    # #HopperOfflineGPUBaseConfig(OfflineGPUBaseConfig):
    # precision = "fp8"
    # vboost_slider = 1
    # trtllm_runtime_flags = {
    #     'kvcache_free_gpu_mem_frac': 0.95,
    #     'enable_chunked_context': True,
    # }
    # trtllm_checkpoint_flags = {
    #     'kv_cache_dtype': 'fp8'
    # }
    # trtllm_build_flags = {
    #     'use_paged_context_fmha': 'enable',
    #     'tokens_per_block': 32,
    #     'tensor_parallelism': 1,
    #     'pipeline_parallelism': 1,
    # }
    # trtllm_runtime_flags = {
    #     'kvcache_free_gpu_mem_frac': 0.90
    # }
    # #H200_SXM_141GBx1(HopperOfflineGPUBaseConfig):
    # gpu_batch_size = {'llama2-70b-interactive': 2048}
    # trtllm_build_flags = {
    #     'max_num_tokens': 1536,
    #     'gemm_swiglu_plugin': 'fp8',
    # }
    # trtllm_runtime_flags = {'max_num_tokens': 1536}
    # offline_expected_qps = 14.4
    # #H200_SXM_141GBx8(H200_SXM_141GBx1):
    # offline_expected_qps = H200_SXM_141GBx1.offline_expected_qps * 8


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class HPE_CRAY_XD670_H200_SXM_141GBX8_HighAccuracy(HPE_CRAY_XD670_H200_SXM_141GBX8):
    pass


