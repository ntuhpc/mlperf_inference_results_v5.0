# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *
''' 
@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class HPE_ProLiant_DL380a_L40S_PCIe_48GBx4(OfflineGPUBaseConfig):
    system = KnownSystem.HPE_ProLiant_DL380a_L40S_PCIe_48GBx4
    gpu_batch_size = {'mixtral-8x7b': 512}
    precision = "fp16"
    use_fp8 = True
    offline_expected_qps = 10 * 4
    enable_sort = False
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    tensor_parallelism = 2 #NVIDIA README says TP=2 for 40GB and lower  GPU memory
    pipeline_parallelism = 1
    kvcache_free_gpu_mem_frac = 0.90
    #min_duration = 2400000

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class HPE_ProLiant_DL380a_H100_PCIe_80GBx4(OfflineGPUBaseConfig):
    system = KnownSystem.HPE_ProLiant_DL380a_H100_PCIe_80GBx4
    gpu_batch_size = {'mixtral-8x7b': 896}
    precision = "fp16"
    use_fp8 = True
    offline_expected_qps = 15 * 4
    enable_sort = False
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    tensor_parallelism = 1  #NVIDIA README says TP=1 for 80GB and higher GPU memory
    pipeline_parallelism = 1
    kvcache_free_gpu_mem_frac = 0.90
    #min_duration = 2400000

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP) #
class HPE_PROLIANT_DL380A_H100_NVL_94GBX4(OfflineGPUBaseConfig):
    system = KnownSystem.HPE_ProLiant_DL380a_H100_NVL_94GBx4
    gpu_batch_size = {'mixtral-8x7b': 896}
    precision = "fp16"
    use_fp8 = True
    offline_expected_qps = 15 * 4
    enable_sort = False
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    tensor_parallelism = 1 #NVIDIA README says TP=1 for 80GB and higher GPU memory
    pipeline_parallelism = 1
    kvcache_free_gpu_mem_frac = 0.90
    #min_duration = 2400000
 '''
@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class HPE_CRAY_XD670_H100_SXM_80GBX8(H100_SXM_80GBx8):
    system = KnownSystem.HPE_Cray_XD670_H100_SXM_80GBx8
    gpu_batch_size = {'mixtral-8x7b': 896} #1024
    precision = "fp8"
    vboost_slider = 1
    trtllm_runtime_flags = {
        'kvcache_free_gpu_mem_frac': 0.90,
        'enable_chunked_context': False,
    }
    # precision = "fp16"
    # use_fp8 = True
    offline_expected_qps = 46 * 8 #48*8
    # enable_sort = False
    # max_num_tokens = 8192
    # vboost_slider = 1
    # gpu_copy_streams = 1
    # gpu_inference_streams = 1
    # tensor_parallelism = 1  #NVIDIA README says TP=1 for 80GB and higher GPU memory
    # pipeline_parallelism = 1
    # kvcache_free_gpu_mem_frac = 0.90
    # min_duration = 2400000

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
    #     'kvcache_free_gpu_mem_frac': 0.90,
    #     'enable_chunked_context': True,
    # }
    # trtllm_checkpoint_flags = {
    #     'kv_cache_dtype': 'fp8',
    #     'effective_bits': 8.75,
    #     'num_calib_steps': 16,
    # }
    # trtllm_build_flags = {
    #     'tokens_per_block': 32,
    #     'tensor_parallelism': 1,
    #     'pipeline_parallelism': 1,
    #     'max_num_tokens': 16 * 1024,
    # }
    # #H100_SXM_80GBx1(HopperOfflineGPUBaseConfig):
    # gpu_batch_size = {'mixtral-8x7b': 896}
    # offline_expected_qps = 46
    # trtllm_build_flags = {'max_num_tokens': 8192}
    # trtllm_runtime_flags = {'max_num_tokens': 8192}
    # #H100_SXM_80GBx8(H100_SXM_80GBx1):
    # offline_expected_qps = 46 * 8

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class HPE_CRAY_XD670_H200_SXM_141GBX8(H200_SXM_141GBx8):
    system = KnownSystem.HPE_Cray_XD670_H200_SXM_141GBx8
    gpu_batch_size = {'mixtral-8x7b': 3072}
    precision = "fp8"
    vboost_slider = 1
    trtllm_runtime_flags = {
        'kvcache_free_gpu_mem_frac': 0.90,
        'enable_chunked_context': False,
    }
    # precision = "fp16"
    # use_fp8 = True
    offline_expected_qps = 58*8 #56 * 8 * 0.97
    # enable_sort = False
    # max_num_tokens = 8192
    # vboost_slider = 1
    # gpu_copy_streams = 1
    # gpu_inference_streams = 1
    # tensor_parallelism = 1  #NVIDIA README says TP=1 for 80GB and higher GPU memory
    # pipeline_parallelism = 1
    # kvcache_free_gpu_mem_frac = 0.90
    # min_duration = 2400000

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
    #     'kvcache_free_gpu_mem_frac': 0.90,
    #     'enable_chunked_context': True,
    # }
    # trtllm_checkpoint_flags = {
    #     'kv_cache_dtype': 'fp8',
    #     'effective_bits': 8.75,
    #     'num_calib_steps': 16,
    # }
    # trtllm_build_flags = {
    #     'tokens_per_block': 32,
    #     'tensor_parallelism': 1,
    #     'pipeline_parallelism': 1,
    #     'max_num_tokens': 16 * 1024,
    # }
    # #H200_SXM_141GBx1(HopperOfflineGPUBaseConfig):
    # gpu_batch_size = {'mixtral-8x7b': 3072}
    # offline_expected_qps = 56
    # trtllm_runtime_flags = {
    #     'max_num_tokens': 9 * 1024,
    # }
    # #H200_SXM_141GBx8(H200_SXM_141GBx1):
    # offline_expected_qps = 56 * 8 * 0.97

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class HPE_PROLIANT_DL380A_H200_NVL_141GBX8(HopperOfflineGPUBaseConfig):
    system = KnownSystem.HPE_PROLIANT_DL380A_H200_NVL_141GBX8
    gpu_batch_size = {'mixtral-8x7b': 3072}
    offline_expected_qps = 56 * 8 * 0.97
    trtllm_build_flags = {'max_num_token': 16384}
    trtllm_runtime_flags = {
        'max_num_tokens':  16384,
    }
    vboost_slider = 1

