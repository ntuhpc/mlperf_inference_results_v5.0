# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *
'''
@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class HPE_ProLiant_DL380a_L40S_PCIe_48GBx4(ServerGPUBaseConfig):
    system = KnownSystem.HPE_ProLiant_DL380a_L40S_PCIe_48GBx4
    gpu_batch_size = {'gptj': 256}
    use_fp8 = True
    server_target_qps = 0.88*7*4

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class HPE_ProLiant_DL380a_L40S_PCIe_48GBx4_HighAccuracy(HPE_ProLiant_DL380a_L40S_PCIe_48GBx4):
    pass

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class HPE_ProLiant_DL380a_H100_PCIe_80GBx4(H100_PCIe_80GBx1):
    system = KnownSystem.HPE_ProLiant_DL380a_H100_PCIe_80GBx4
    gpu_batch_size = {'gptj': 128}
    server_target_qps = 16*4

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class HPE_ProLiant_DL380a_H100_PCIe_80GBx4_HighAccuracy(HPE_ProLiant_DL380a_H100_PCIe_80GBx4):
    pass

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class HPE_PROLIANT_DL380A_H100_NVL_94GBX4(ServerGPUBaseConfig):
    system = KnownSystem.HPE_ProLiant_DL380a_H100_NVL_94GBx4
    gpu_batch_size = {'gptj': 128}
    use_fp8 = True
    server_target_qps = 23.5 * 4 #_valid

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class HPE_PROLIANT_DL380A_H100_NVL_94GBX4_HighAccuracy(HPE_PROLIANT_DL380A_H100_NVL_94GBX4):
    pass
'''
@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class HPE_CRAY_XD670_H100_SXM_80GBX8(H100_SXM_80GBx8):
    system = KnownSystem.HPE_Cray_XD670_H100_SXM_80GBx8
    gpu_batch_size = {'gptj': 256}
    precision = "fp8"
    #use_fp8 = True
    server_target_qps = 38*8 #36*8
    vboost_slider = 1 #<<new

    #ServerGPUBaseConfig(GPUBaseConfig):
    enable_sort = False
    trtllm_runtime_flags = {
        'batch_scheduler_policy': 'max_util',
        'context_chunking_policy': 'first_come_first_served',
    }
    trtllm_build_flags = {
        'tensor_parallelism': 1,
        'pipeline_parallelism': 1,
    }
    #HopperServerGPUBaseConfig(ServerGPUBaseConfig):
    #precision = "fp8"
    #vboost_slider = 1
    #H100_SXM_80GBx1(HopperServerGPUBaseConfig):
    #gpu_batch_size = {'gptj': 256}
    #server_target_qps = 34.92
    #H100_SXM_80GBx8(H100_SXM_80GBx1):
    #server_target_qps = 34.92 * 8

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class HPE_CRAY_XD670_H100_SXM_80GBX8_HighAccuracy(HPE_CRAY_XD670_H100_SXM_80GBX8):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class HPE_CRAY_XD670_H200_SXM_141GBX8(H200_SXM_141GBx8):
    system = KnownSystem.HPE_Cray_XD670_H200_SXM_141GBx8
    gpu_batch_size = {'gptj': 396}
    #precision = "fp8"
    use_fp8 = True
    server_target_qps = 38 * 8 #40 * 8
    vboost_slider = 1

    #ServerGPUBaseConfig(GPUBaseConfig):
    enable_sort = False
    trtllm_runtime_flags = {
        'batch_scheduler_policy': 'max_util',
        'context_chunking_policy': 'first_come_first_served',
    }
    trtllm_build_flags = {
        'tensor_parallelism': 1,
        'pipeline_parallelism': 1,
    }
    #HopperServerGPUBaseConfig(ServerGPUBaseConfig):
    #precision = "fp8"
    #vboost_slider = 1
    #H200_SXM_141GBx1(HopperServerGPUBaseConfig):
    #gpu_batch_size = {'gptj': 396}
    #server_target_qps = 40
    #H200_SXM_141GBx8(H200_SXM_141GBx1):
    #server_target_qps = 40 * 8

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class HPE_CRAY_XD670_H200_SXM_141GBX8_HighAccuracy(HPE_CRAY_XD670_H200_SXM_141GBX8):
    pass

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class HPE_PROLIANT_DL380A_L40S_PCIE_48GBX8(ServerGPUBaseConfig):
    system = KnownSystem.HPE_ProLiant_DL380a_L40S_PCIe_48GBx8
    gpu_batch_size = {'gptj': 128}
    use_fp8 = True
    server_target_qps = 100
    precision = 'fp8'
    use_token_latencies = False
    tensor_path = 'build/models/gpt-j/fp8-quantized-ammo/GPTJ-FP8-quantized'


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class HPE_PROLIANT_DL380A_L40S_PCIE_48GBX8_HighAccuracy(HPE_PROLIANT_DL380A_L40S_PCIE_48GBX8):
    pass

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class HPE_PROLIANT_DL380A_H200_NVL_141GBX8(HopperServerGPUBaseConfig):
    system = KnownSystem.HPE_PROLIANT_DL380A_H200_NVL_141GBX8
    gpu_batch_size = {'gptj': 196}
    use_fp8 = True
    enable_sort = False
    server_target_qps = 265
    vboost_slider = 1

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class HPE_PROLIANT_DL380A_H200_NVL_141GBX8_HighAccuracy(HPE_PROLIANT_DL380A_H200_NVL_141GBX8):
    pass

"""
    # Applicable fields for this benchmark are listed below. Not all of these are necessary, and some may be defined in the BaseConfig already and inherited.
    # Please see NVIDIA's submission config files for example values and which fields to keep.
    # Required fields (Must be set or inherited to run):
    gpu_batch_size: Dict = {}
    input_dtype: str = ''
    input_format: str = ''
    precision: str = ''
    tensor_path: str = ''

    # Optional fields:
    active_sms: int = 0
    buffer_manager_thread_count: int = 0
    cache_file: str = ''
    coalesced_tensor: bool = False
    enable_sort: bool = False
    gpu_copy_streams: int = 0
    gpu_inference_streams: int = 0
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
    server_target_qps: float = 0.0
    server_target_qps_adj_factor: float = 0.0
    tensor_parallelism: int = 0
    trtllm_batch_sched_policy: str = ''
    trtllm_batching_mode: str = ''
    use_fp8: bool = False
    use_graphs: bool = False
    use_jemalloc: bool = False
    use_spin_wait: bool = False
    use_token_latencies: bool = False
    vboost_slider: int = 0
    verbose_glog: int = 0
    workspace_size: int = 0
"""


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class HPE_HAWKS_3200_H100_NVL_94GBX4(ServerGPUBaseConfig):
    system = KnownSystem.HPE_Hawks_3200_H100_NVL_94GBx4

    # Applicable fields for this benchmark are listed below. Not all of these are necessary, and some may be defined in the BaseConfig already and inherited.
    # Please see NVIDIA's submission config files for example values and which fields to keep.
    # Required fields (Must be set or inherited to run):
    gpu_batch_size = {'gptj': 256}
    precision = "fp8"
    #use_fp8 = True
    server_target_qps = 26*4 #34.92*8
    vboost_slider = 1 #<<new

    #ServerGPUBaseConfig(GPUBaseConfig):
    enable_sort = False
    trtllm_runtime_flags = {
        'batch_scheduler_policy': 'max_util',
        'context_chunking_policy': 'first_come_first_served',
    }
    trtllm_build_flags = {
        'tensor_parallelism': 1,
        'pipeline_parallelism': 1,
    }

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class HPE_HAWKS_3200_H100_NVL_94GBX4_HighAccuracy(HPE_HAWKS_3200_H100_NVL_94GBX4):
    server_target_qps = 24*4


