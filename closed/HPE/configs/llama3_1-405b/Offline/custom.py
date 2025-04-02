# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.json

from . import *

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class HPE_CRAY_XD670_H100_SXM_80GBX8(H200_SXM_141GBx8_TP4PP2):
    system = KnownSystem.HPE_Cray_XD670_H100_SXM_80GBx8
    offline_expected_qps = 0.6

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class HPE_CRAY_XD670_H200_SXM_141GBX8(H200_SXM_141GBx8_TP4PP2):
    system = KnownSystem.HPE_Cray_XD670_H200_SXM_141GBx8
    offline_expected_qps = 0.99 #0.9
    # #OfflineGPUBaseConfig(GPUBaseConfig):
    # min_duration = 600000
    # enable_sort = False
    # trtllm_build_flags = {
    # }
    # trtllm_runtime_flags = {
    #     'batch_scheduler_policy': 'max_util',
    #     'context_chunking_policy': 'first_come_first_served',
    # }
    # #HopperOfflineGPUBaseConfig(OfflineGPUBaseConfig):
    # precision = "fp8"
    # vboost_slider = 1
    trtllm_runtime_flags = {
        'kvcache_free_gpu_mem_frac': 0.96,
        'enable_chunked_context': True,
    }
    # trtllm_checkpoint_flags = {
    #     'kv_cache_dtype': 'fp8'
    # }
    # trtllm_build_flags = {
    #     'use_paged_context_fmha': 'enable',
    #     'tokens_per_block': 32,
    #     'use_fp8_context_fmha': 'enable',
    # }
    # #H200_SXM_141GBx8_TP4PP2(HopperOfflineGPUBaseConfig):
    # offline_expected_qps = 0.9
    # trtllm_build_flags = {
    #     'max_num_tokens': 2560,
    #     'tensor_parallelism': 4,
    #     'pipeline_parallelism': 2,
    #     # Disable to prevent intermittent failures
    #     'gemm_allreduce_plugin': 'float16',
    # }
    # trtllm_runtime_flags = {'max_num_tokens': 1536}
    # gpu_batch_size = {'llama3.1-405b': 512}


'''     # Applicable fields for this benchmark are listed below. Not all of these are necessary, and some may be defined in the BaseConfig already and inherited.
    # Please see NVIDIA's submission config files for example values and which fields to keep.
    # Required fields (Must be set or inherited to run):
    gpu_batch_size: Dict = {}
    tensor_path: str = ''
    trtllm_build_flags: parse_cli_flags = {}
    trtllm_checkpoint_flags: parse_cli_flags = {}
    trtllm_runtime_flags: parse_cli_flags = {}

    # Optional fields:
    active_sms: int = 0
    cache_file: str = ''
    checkpoint_dir: str = ''
    enable_sort: bool = False
    engine_dir: str = ''
    llm_gen_config_path: str = ''
    offline_expected_qps: float = 0.0
    use_token_latencies: bool = False
    vboost_slider: int = 0
    workspace_size: int = 0

 '''
