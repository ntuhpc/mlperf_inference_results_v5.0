# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *
''' 
@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class HPE_ProLiant_DL380a_L40S_PCIe_48GBx4(ServerGPUBaseConfig):
    system = KnownSystem.HPE_ProLiant_DL380a_L40S_PCIe_48GBx4
    gpu_batch_size = {'mixtral-8x7b': 192}
    precision = "fp16"
    use_fp8 = True
    enable_sort = False
    server_target_qps = 8
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    tensor_parallelism = 2  #NVIDIA README says TP=2 for 40GB and lower GPU memory
    pipeline_parallelism = 1
    kvcache_free_gpu_mem_frac = 0.90
    #min_duration = 2400000

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class HPE_ProLiant_DL380a_H100_PCIe_80GBx4(ServerGPUBaseConfig):
    system = KnownSystem.HPE_ProLiant_DL380a_H100_PCIe_80GBx4
    gpu_batch_size = {'mixtral-8x7b': 1024}
    precision = "fp16"
    use_fp8 = True
    enable_sort = False
    server_target_qps = 18.4
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    tensor_parallelism = 1 #NVIDIA README says TP=1 for 80GB and higher GPU memory
    pipeline_parallelism = 1
    kvcache_free_gpu_mem_frac = 0.90
    #min_duration = 2400000

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class HPE_PROLIANT_DL380A_H100_NVL_94GBX4(ServerGPUBaseConfig):
    system = KnownSystem.HPE_ProLiant_DL380a_H100_NVL_94GBx4
    gpu_batch_size = {'mixtral-8x7b': 1024}
    precision = "fp16"
    use_fp8 = True
    enable_sort = False
    server_target_qps = 20 * 4
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
    num_calib_steps=16
    precision = 'fp8'
    vboost_slider = 1
    trtllm_runtime_flags = {
        'kvcache_free_gpu_mem_frac': 0.90,
        'enable_chunked_context': False,
    }
    # gpu_batch_size = {'mixtral-8x7b': 896} #1024
    # precision = "fp16"
    # use_fp8 = True
    server_target_qps = 45 * 8 #43.5 * 8
    # enable_sort = False
    # vboost_slider = 1
    # gpu_copy_streams = 1
    # gpu_inference_streams = 1 #2
    # tensor_parallelism = 1 #NVIDIA README says TP=1 for 80GB and higher GPU memory
    # pipeline_parallelism = 1
    # kvcache_free_gpu_mem_frac = 0.90 #0.90
    # min_duration = 2400000

    # #ServerGPUBaseConfig(GPUBaseConfig):
    # enable_sort = False
    # min_duration = 2400000
    # trtllm_runtime_flags = {
    #     'batch_scheduler_policy': 'max_util',
    #     'context_chunking_policy': 'first_come_first_served',
    # }
    # #HopperServerGPUBaseConfig(ServerGPUBaseConfig):
    # precision = 'fp8'
    # vboost_slider = 1
    # trtllm_runtime_flags = {
    #     'kvcache_free_gpu_mem_frac': 0.90,
    #     'enable_chunked_context': True,
    # }
    # trtllm_checkpoint_flags = {
    #     'kv_cache_dtype': 'fp8',
    #     'effective_bits': 8.75
    # }
    # trtllm_build_flags = {
    #     'tokens_per_block': 32,
    #     'tensor_parallelism': 1,
    #     'pipeline_parallelism': 1,
    # }
    # #H100_SXM_80GBx1(HopperServerGPUBaseConfig):
    # gpu_batch_size = {'mixtral-8x7b': 896}
    # server_target_qps = 43.5
    # #H100_SXM_80GBx8(H100_SXM_80GBx1):
    # server_target_qps = 43.5 * 8



@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class HPE_CRAY_XD670_H200_SXM_141GBX8(H200_SXM_141GBx8):
    system = KnownSystem.HPE_Cray_XD670_H200_SXM_141GBx8
    num_calib_steps=16
    precision = 'fp8'
    vboost_slider = 1
    trtllm_runtime_flags = {
        'kvcache_free_gpu_mem_frac': 0.93,
        'enable_chunked_context': False,
    }
    # gpu_batch_size = {'mixtral-8x7b': 1200}
    # precision = "fp16"
    # use_fp8 = True
    server_target_qps = 52.5*8 #52*8
    # enable_sort = False
    # vboost_slider = 1
    # max_num_tokens = 8192
    # gpu_copy_streams = 1
    # gpu_inference_streams = 1 #2
    # tensor_parallelism = 1 #NVIDIA README says TP=1 for 80GB and higher GPU memory
    # pipeline_parallelism = 1
    # kvcache_free_gpu_mem_frac = 0.95 #0.90
    # min_duration = 2400000

    # #ServerGPUBaseConfig(GPUBaseConfig):
    # enable_sort = False
    # min_duration = 2400000
    # trtllm_runtime_flags = {
    #     'batch_scheduler_policy': 'max_util',
    #     'context_chunking_policy': 'first_come_first_served',
    # }
    # #HopperServerGPUBaseConfig(ServerGPUBaseConfig):
    # precision = 'fp8'
    # vboost_slider = 1
    # trtllm_runtime_flags = {
    #     'kvcache_free_gpu_mem_frac': 0.90,
    #     'enable_chunked_context': True,
    # }
    # trtllm_checkpoint_flags = {
    #     'kv_cache_dtype': 'fp8',
    #     'effective_bits': 8.75
    # }
    # trtllm_build_flags = {
    #     'tokens_per_block': 32,
    #     'tensor_parallelism': 1,
    #     'pipeline_parallelism': 1,
    # }
    # #H200_SXM_141GBx1(HopperServerGPUBaseConfig):
    # trtllm_build_flags = {'max_num_tokens': 8192}
    # trtllm_runtime_flags = {'max_num_tokens': 8192}
    # gpu_batch_size = {'mixtral-8x7b': 3072}
    # server_target_qps = 53.5 # 52, 53 works
    # #H200_SXM_141GBx8(H200_SXM_141GBx1):
    # server_target_qps = H200_SXM_141GBx1.server_target_qps * 8
    
@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class HPE_PROLIANT_DL380A_L40S_PCIE_48GBX8(ServerGPUBaseConfig):
    system = KnownSystem.HPE_ProLiant_DL380a_L40S_PCIe_48GBx8
    
    gpu_batch_size = {'mixtral-8x7b': 512}
    server_target_qps = 43
    use_fp8 = True
    precision = 'fp8'
    trtllm_build_flags = {
        'tensor_parallelism': 2,
        'pipeline_parallelism': 1
    }
    trtllm_checkpoint_flags = {
        'num_calib_steps': 16
    }
    # trtllm_runtime_flags = {
    #     'enable_chunked_context': True
    # }

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class HPE_PROLIANT_DL380A_H200_NVL_141GBX8(HopperServerGPUBaseConfig):
    system = KnownSystem.HPE_PROLIANT_DL380A_H200_NVL_141GBX8
    gpu_batch_size = {'mixtral-8x7b': 3072}
    trtllm_build_flags = {'max_num_tokens': 8192}
    trtllm_runtime_flags = {'max_num_tokens': 8192}
    server_target_qps = 344

