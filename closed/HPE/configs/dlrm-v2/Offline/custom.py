# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class HPE_CRAY_XD670_H100_SXM_80GBX8(H100_SXM_80GBx8):
    system = KnownSystem.HPE_Cray_XD670_H100_SXM_80GBx8
    check_contiguity = True
    offline_expected_qps = 80000 * 8 #77000 * 8
    gpu_batch_size = {'dlrm-v2': 600_000} #60000
    embedding_weights_on_gpu_part: float = 1.0
    vboost_slider = 1
    #start_from_device = True
    #numa_config = "0-3:0-47,96-143&4-7:48-95,144-191"
    numa_config = "0-1:0-27,112-139&2-3:28-55,140-167&4-5:56-83,168-195&6-7:84-111,196-223"

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class HPE_CRAY_XD670_H100_SXM_80GBX8_HighAccuracy(HPE_CRAY_XD670_H100_SXM_80GBX8):
    offline_expected_qps = 49000 * 8 #47500 * 8 #44000 * 8
    interaction_op_precision = 'fp16'

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class HPE_CRAY_XD670_H200_SXM_141GBX8(H200_SXM_141GBx8):
    system = KnownSystem.HPE_Cray_XD670_H200_SXM_141GBx8
    #check_contiguity = True
    gpu_batch_size = {'dlrm-v2': 600_000} #60000
    embedding_weights_on_gpu_part: float = 1.0
    offline_expected_qps = 94000 * 8 #90000 * 8
    start_from_device = True
    vboost_slider = 1
    #numa_config = "0-3:0-31,64-95&4-7:32-63,96-127"

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class HPE_CRAY_XD670_H200_SXM_141GBX8_HighAccuracy(HPE_CRAY_XD670_H200_SXM_141GBX8):
    offline_expected_qps = 52000 * 8 #51000 * 8
    interaction_op_precision = 'fp16'

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class HPE_PROLIANT_DL380A_H200_NVL_141GBX8(OfflineGPUBaseConfig):
    system = KnownSystem.HPE_PROLIANT_DL380A_H200_NVL_141GBX8
    gpu_batch_size = {'dlrm-v2': 60000}
    embedding_weights_on_gpu_part: float = 1.0
    offline_expected_qps = 90000 * 8
    start_from_device = True
    vboost_slider = 1

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class HPE_PROLIANT_DL380A_H200_NVL_141GBX8_HighAccuracy(HPE_PROLIANT_DL380A_H200_NVL_141GBX8):
    offline_expected_qps = 51000 * 8
    interaction_op_precision = 'fp16'

    # Applicable fields for this benchmark are listed below. Not all of these are necessary, and some may be defined in the BaseConfig already and inherited.
    # Please see NVIDIA's submission config files for example values and which fields to keep.
    # Required fields (Must be set or inherited to run):

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class HPE_ProLiant_DL380a_L40S_PCIe_48GBx8(OfflineGPUBaseConfig):
    system = KnownSystem.HPE_ProLiant_DL380a_L40S_PCIe_48GBx8
    embedding_weights_on_gpu_part: float = 1.0
    gpu_batch_size = {'dlrm-v2': 7500}
    offline_expected_qps = 25000*12

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class HPE_ProLiant_DL380a_L40S_PCIe_48GBx8_HighAccuracy(HPE_ProLiant_DL380a_L40S_PCIe_48GBx8):
    offline_expected_qps = 98000
    interaction_op_precision = 'fp16'

'''
# Optional fields:
    active_sms: int = 0
    batch_triton_requests: bool = False
    buffer_manager_thread_count: int = 0
    cache_file: str = ''
    check_contiguity: bool = False
    coalesced_tensor: bool = False
    complete_threads: int = 0
    embedding_weights_on_gpu_part: float = 0.0
    gather_kernel_buffer_threshold: int = 0
    gpu_copy_streams: int = 0
    gpu_inference_streams: int = 0
    gpu_num_bundles: int = 0
    instance_group_count: int = 0
    max_pairs_per_staging_thread: int = 0
    max_queue_delay_usec: int = 0
    mega_table_npy_file: str = ''
    mega_table_scales_npy_file: str = ''
    model_path: str = ''
    num_concurrent_batchers: int = 0
    num_concurrent_issuers: int = 0
    num_staging_batches: int = 0
    num_staging_threads: int = 0
    numa_config: str = ''
    offline_expected_qps: float = 0.0
    output_pinned_memory: bool = False
    performance_sample_count_override: int = 0
    preferred_batch_size: str = ''
    qsl_numa_override: str = ''
    reduced_precision_io: int = 0
    request_timeout_usec: int = 0
    row_frequencies_npy_filepath: str = ''
    run_infer_on_copy_streams: bool = False
    sample_partition_path: str = ''
    use_batcher_thread_per_device: bool = False
    use_concurrent_harness: bool = False
    use_graphs: bool = False
    use_jemalloc: bool = False
    use_spin_wait: bool = False
    verbose_glog: int = 0
    warmup_duration: float = 0.0
    workspace_size: int = 0
'''


