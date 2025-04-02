# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *

'''
@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class HPE_ProLiant_DL380a_L40S_PCIe_48GBx4(ServerGPUBaseConfig):
    server_target_qps = 16.7*4
    system = KnownSystem.HPE_ProLiant_DL380a_L40S_PCIe_48GBx4
    gpu_batch_size = {'llama2-70b': 256}
    use_fp8 = True
    enable_sort = False

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class HPE_ProLiant_DL380a_L40S_PCIe_48GBx4_HighAccuracy(HPE_ProLiant_DL380a_L40S_PCIe_48GBx4):
    pass

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class HPE_ProLiant_DL380a_H100_PCIe_80GBx4(ServerGPUBaseConfig):
    system = KnownSystem.HPE_ProLiant_DL380a_H100_PCIe_80GBx4
    server_target_qps = 5*4
    gpu_batch_size = {'llama2-70b': 1024}
    kvcache_free_gpu_mem_frac = 0.95 #0.90
    use_fp8 = True
    enable_sort = False
    tensor_parallelism = 2

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class HPE_ProLiant_DL380a_H100_PCIe_80GBx4_HighAccuracy(HPE_ProLiant_DL380a_H100_PCIe_80GBx4):
    pass

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class HPE_PROLIANT_DL380A_H100_NVL_94GBX4(H100_NVL_94GB_TP2x2):
    system = KnownSystem.HPE_ProLiant_DL380a_H100_NVL_94GBx4
    server_target_qps = 14*2
    gpu_batch_size = {'llama2-70b': 1024}
    kvcache_free_gpu_mem_frac = 0.95 #0.90
    use_fp8 = True
    enable_sort = False
    tensor_parallelism = 2
    vboost_slider = 1

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class HPE_PROLIANT_DL380A_H100_NVL_94GBX4_HighAccuracy(HPE_PROLIANT_DL380A_H100_NVL_94GBX4):
    pass
 '''

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class HPE_CRAY_XD670_H100_SXM_80GBX8(H100_SXM_80GB_TP2x4):
    system = KnownSystem.HPE_Cray_XD670_H100_SXM_80GBx8
    server_target_qps = 82 #80
    gpu_batch_size = {'llama2-70b': 2048} #1048
    kvcache_free_gpu_mem_frac = 0.97 #0.90
    precision = "fp8"
    #use_fp8 = True
    #enable_sort = False
    #tensor_parallelism = 2
    #vboost_slider = 1

    # #ServerGPUBaseConfig(GPUBaseConfig):
    # min_duration = 2400000
    # enable_sort = False
    # #HopperServerGPUBaseConfig(ServerGPUBaseConfig):
    # precision = "fp8"
    # vboost_slider = 1
    # trtllm_checkpoint_flags = {
    #     'kv_cache_dtype': 'fp8'
    # }
    # trtllm_build_flags = {
    #     'tensor_parallelism': 1,
    #     'pipeline_parallelism': 1,
    # }
    # #H100_SXM_80GB_TP2x1(HopperServerGPUBaseConfig):
    # gpu_batch_size = {'llama2-70b': 2048}
    # server_target_qps = 13.533
    # trtllm_build_flags = {
    #     'tensor_parallelism': 2,
    #     'pipeline_parallelism': 1,
    #     'gemm_swiglu_plugin': 'fp8',
    # }
    # #H100_SXM_80GB_TP2x2(H100_SXM_80GB_TP2x1):
    # server_target_qps = 18.4 * 2
    # #H100_SXM_80GB_TP2x4(H100_SXM_80GB_TP2x2):
    # server_target_qps = 75

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class HPE_CRAY_XD670_H100_SXM_80GBX8_HighAccuracy(HPE_CRAY_XD670_H100_SXM_80GBX8):
    #server_target_qps = 80 #20 * 4 #66 #13.5 * 4  #63
    #gpu_batch_size = {'llama2-70b': 1024}
    #kvcache_free_gpu_mem_frac = 0.97 #0.90
    #use_fp8 = True
    pass

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class HPE_CRAY_XD670_H200_SXM_141GBX8(H200_SXM_141GBx8):
    system = KnownSystem.HPE_Cray_XD670_H200_SXM_141GBx8
    #server_target_qps = 105 #103
    #gpu_batch_size = {'llama2-70b': 2048} #896 #1024 #<== changed this back to 896
    #kvcache_free_gpu_mem_frac = 0.97 #0.90
    #precision = "fp8"
    gpu_batch_size = {'llama2-70b': 896}
    server_target_qps = 108 #105
    use_fp8 = True
    enable_sort = False
    tensor_parallelism = 1
    vboost_slider = 1

    # #ServerGPUBaseConfig(GPUBaseConfig):
    # min_duration = 2400000
    # enable_sort = False
    # #HopperServerGPUBaseConfig(ServerGPUBaseConfig):
    # precision = "fp8"
    # vboost_slider = 1
    # trtllm_checkpoint_flags = {
    #     'kv_cache_dtype': 'fp8'
    # }
    # trtllm_build_flags = {
    #     'tensor_parallelism': 1,
    #     'pipeline_parallelism': 1,
    # }
    # #H200_SXM_141GBx1(HopperServerGPUBaseConfig):
    # gpu_batch_size = {'llama2-70b': 2048}
    # server_target_qps = 14.28
    # trtllm_build_flags = {
    #     'max_num_tokens': 1536,
    #     'gemm_swiglu_plugin': 'fp8',
    # }
    # trtllm_runtime_flags = {'max_num_tokens': 1536}
    # #H200_SXM_141GBx8(H200_SXM_141GBx1):
    # server_target_qps = 102.5

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class HPE_CRAY_XD670_H200_SXM_141GBX8_HighAccuracy(HPE_CRAY_XD670_H200_SXM_141GBX8):
    pass

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class HPE_PROLIANT_DL380A_L40S_PCIE_48GBX8(ServerGPUBaseConfig):
    system = KnownSystem.HPE_ProLiant_DL380a_L40S_PCIe_48GBx8
    gpu_batch_size = {"llama2-70b": 512}
    server_target_qps = 6

    # trtllm_runtime_flags = {
    #     'max_batch_size': 1024,
    #     'exclude_input_from_output': True,
    #     'use_inflight_batching': True,
    #     'max_num_tokens': 2048,
    #     'enable_chunked_context': False,
    #     'kvcache_free_gpu_mem_frac': 0.70
    # }
    # trtllm_build_flags = {
    #     'tensor_parallelism': 2,
    #     'pipeline_parallelism': 1,
    #     'max_beam_width': 4,
    #     'kv_cache_type': 'paged',
    #     'remove_input_padding': 'enable',
    #     'multiple_profiles': 'enable',
    #     'use_fused_mlp': 'enable',
    #     'context_fmha': 'enable',
    #     'max_num_tokens': 2048,
    #     'max_input_len': 1024,
    #     'max_seq_len': 1024 + 1024,
    #     'tokens_per_block': 64,
    #     'use_fp8_context_fmha': 'disable',
    #     'use_paged_context_fmha': 'disable',
    # }


    trtllm_build_flags = {
        'tensor_parallelism': 2,
        'pipeline_parallelism': 1,
        # 'tokens_per_block': 64
    }
    trtllm_runtime_flags = {
        'max_batch_size': 512,
        'batch_scheduler_policy': 'no_evict'
    }
    use_fp8 = True
    precision = 'fp8'


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class HPE_PROLIANT_DL380A_L40S_PCIE_48GBX8_HighAccuracy(HPE_PROLIANT_DL380A_L40S_PCIE_48GBX8):
    pass

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class HPE_PROLIANT_DL380A_H200_NVL_141GBX8(HopperServerGPUBaseConfig):
    system = KnownSystem.HPE_PROLIANT_DL380A_H200_NVL_141GBX8
    gpu_batch_size = {'llama2-70b': 2048}
    trtllm_build_flags = {
        'max_num_tokens': 1536,
        'gemm_swiglu_plugin': 'fp8',
    }
    trtllm_runtime_flags = {'max_num_tokens': 1536}
    server_target_qps = 95.5

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
    performance_sample_count_override: int = 0
    pipeline_parallelism: int = 0
    preferred_batch_size: str = ''
    request_timeout_usec: int = 0
    run_infer_on_copy_streams: bool = False
    schedule_rng_seed: int = 0
    server_num_issue_query_threads: int = 0
    server_target_latency_ns: int = 0
    server_target_latency_percentile: float = 0.0
    server_target_qps: int = 0
    server_target_qps_adj_factor: float = 0.0
    tensor_parallelism: int = 0
    use_fp8: bool = False
    use_graphs: bool = False
    use_jemalloc: bool = False
    use_spin_wait: bool = False
    use_token_latencies: bool = False
    verbose_glog: int = 0
    workspace_size: int = 0
 """
