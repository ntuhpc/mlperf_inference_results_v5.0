# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *

''' 
@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class HPE_ProLiant_DL380a_L40S_PCIe_48GBx4(OfflineGPUBaseConfig):
    system = KnownSystem.HPE_ProLiant_DL380a_L40S_PCIe_48GBx4
    scenario = Scenario.Offline
    gpu_batch_size = {'llama2-70b': 128}
    use_fp8 = True
    offline_expected_qps = 1.25 * 4 
    enable_sort = False

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class HPE_ProLiant_DL380a_L40S_PCIe_48GBx4_HighAccuracy(HPE_ProLiant_DL380a_L40S_PCIe_48GBx4):
    pass

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class HPE_ProLiant_DL380a_H100_PCIe_80GBx4(OfflineGPUBaseConfig):
    system = KnownSystem.HPE_ProLiant_DL380a_H100_PCIe_80GBx4
    offline_expected_qps = 14 * 2 #10 * 2
    gpu_batch_size = {'llama2-70b': 512} #1024
    kvcache_free_gpu_mem_frac = 0.97 #0.95
    use_fp8 = True
    enable_sort = False
    tensor_parallelism = 2

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class HPE_ProLiant_DL380a_H100_PCIe_80GBx4_HighAccuracy(HPE_ProLiant_DL380a_H100_PCIe_80GBx4):
    pass

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class HPE_PROLIANT_DL380A_H100_NVL_94GBX4(H100_NVL_94GB_TP2x2):
    system = KnownSystem.HPE_ProLiant_DL380a_H100_NVL_94GBx4
    offline_expected_qps = 15 * 2
    gpu_batch_size = {'llama2-70b': 1300}
    kvcache_free_gpu_mem_frac = 0.95 #0.95
    use_fp8 = True
    enable_sort = False
    tensor_parallelism = 2
    vboost_slider = 1

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class HPE_PROLIANT_DL380A_H100_NVL_94GBX4_HighAccuracy(HPE_PROLIANT_DL380A_H100_NVL_94GBX4):
    pass
 '''

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class HPE_CRAY_XD670_H100_SXM_80GBX8(H100_SXM_80GB_PP2x4):
    system = KnownSystem.HPE_Cray_XD670_H100_SXM_80GBx8
'''     precision = "fp8"
    #use_fp8 = True
    offline_expected_qps = 20 * 4
    gpu_batch_size = {'llama2-70b': 2048} #896
    kvcache_free_gpu_mem_frac = 0.95 #0.97
    #enable_sort = False
    #tensor_parallelism = 2
    vboost_slider = 1 '''

    # #OfflineGPUBaseConfig(GPUBaseConfig):
    # enable_sort = False
    # min_duration = 2400000
    # #HopperOfflineGPUBaseConfig(OfflineGPUBaseConfig):
    # precision = "fp8"
    # vboost_slider = 1
    # trtllm_checkpoint_flags = {
    #     'kv_cache_dtype': 'fp8'
    # }
    # trtllm_build_flags = {
    #     'tensor_parallelism': 1,
    #     'pipeline_parallelism': 1,
    # }
    #
    # #H100_SXM_80GB_PP2x1(HopperOfflineGPUBaseConfig):
    # gpu_batch_size = {'llama2-70b': 2048}
    # offline_expected_qps = 14.0
    # trtllm_build_flags = {
    #     'max_num_tokens': 1024,
    #     'tensor_parallelism': 1,
    #     'pipeline_parallelism': 2,
    #     'reduce_fusion': 'enable',
    #     'gemm_swiglu_plugin': 'fp8',
    # }
    # trtllm_runtime_flags = {'max_num_tokens': 1024}
    # #H100_SXM_80GB_PP2x2(H100_SXM_80GB_PP2x1):
    # offline_expected_qps = H100_SXM_80GB_PP2x1.offline_expected_qps * 2
    # #H100_SXM_80GB_PP2x4(H100_SXM_80GB_PP2x2):
    # offline_expected_qps = H100_SXM_80GB_PP2x2.offline_expected_qps * 2


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class HPE_CRAY_XD670_H100_SXM_80GBX8_HighAccuracy(HPE_CRAY_XD670_H100_SXM_80GBX8):
    pass

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class HPE_CRAY_XD670_H200_SXM_141GBX8(H200_SXM_141GBx8):
    system = KnownSystem.HPE_Cray_XD670_H200_SXM_141GBx8
    gpu_batch_size = {'llama2-70b': 806} #2048
    #precision = "fp8"
    use_fp8 = True
    offline_expected_qps = 104 #112
    enable_sort = False
    tensor_parallelism = 1
    vboost_slider = 1
    #kvcache_free_gpu_mem_frac = 0.95 #0.97

    # #OfflineGPUBaseConfig(GPUBaseConfig):
    # enable_sort = False
    # min_duration = 2400000
    # #HopperOfflineGPUBaseConfig(OfflineGPUBaseConfig):
    # precision = "fp8"
    # vboost_slider = 1
    # trtllm_checkpoint_flags = {
    #     'kv_cache_dtype': 'fp8'
    # }
    # trtllm_build_flags = {
    #     'tensor_parallelism': 1,
    #     'pipeline_parallelism': 1,
    # }
    # #H200_SXM_141GBx1(HopperOfflineGPUBaseConfig):
    # gpu_batch_size = {'llama2-70b': 2048}
    # offline_expected_qps = 14.4
    # trtllm_build_flags = {
    #     'max_num_tokens': 1536,
    #     'gemm_swiglu_plugin': 'fp8',
    # }
    # trtllm_runtime_flags = {'max_num_tokens': 1536}
    # #H200_SXM_141GBx8(H200_SXM_141GBx1):
    # offline_expected_qps = 104

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class HPE_CRAY_XD670_H200_SXM_141GBX8_HighAccuracy(HPE_CRAY_XD670_H200_SXM_141GBX8):
    pass

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class HPE_PROLIANT_DL380A_H200_NVL_141GBX8(HopperOfflineGPUBaseConfig):
    system = KnownSystem.HPE_PROLIANT_DL380A_H200_NVL_141GBX8
    gpu_batch_size = {'llama2-70b': 2048}
    trtllm_build_flags = {
        'max_num_tokens': 1536,
        'gemm_swiglu_plugin': 'fp8',
    }
    trtllm_runtime_flags = {'max_num_tokens': 1536}
    offline_expected_qps = 104

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class HPE_PROLIANT_DL380A_H200_NVL_141GBX8_HighAccuracy(HPE_PROLIANT_DL380A_H200_NVL_141GBX8):
    pass

"""     # Applicable fields for this benchmark are listed below. Not all of these are necessary, and some may be defined in the BaseConfig already and inherited.
    # Please see NVIDIA's submission config files for example values and which fields to keep.
    # Required fields (Must be set or inherited to run):

    # Optional fields:
    active_sms: int = 0
    buffer_manager_thread_count: int = 0
    cache_file: str = ''
    coalesced_tensor: bool = False
    enable_sort: bool = False
    gpu_copy_streams: int = 0
    gpu_inference_streams: int = 0
    gpu_rank_map: str = ''
    instance_group_count: int = 0
    kvcache_free_gpu_mem_frac: float = 0.0
    llm_gen_config_path: str = ''
    max_num_tokens: int = 0
    model_path: str = ''
    num_sort_segments: int = 0
    numa_config: str = ''
    offline_expected_qps: float = 0.0
    performance_sample_count_override: int = 0
    pipeline_parallelism: int = 0
    preferred_batch_size: str = ''
    request_timeout_usec: int = 0
    run_infer_on_copy_streams: bool = False
    tensor_parallelism: int = 0
    use_fp8: bool = False
    use_graphs: bool = False
    use_jemalloc: bool = False
    use_spin_wait: bool = False
    use_token_latencies: bool = False
    verbose_glog: int = 0
    workspace_size: int = 0
 """


