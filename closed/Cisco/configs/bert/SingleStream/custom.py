# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.json

from . import *


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class C245M8_H100NVL_94GBX2(SingleStreamGPUBaseConfig):
    system = KnownSystem.C245M8_H100NVL_94GBx2

    # Applicable fields for this benchmark are listed below. Not all of these are necessary, and some may be defined in the BaseConfig already and inherited.
    # Please see NVIDIA's submission config files for example values and which fields to keep.
    # Required fields (Must be set or inherited to run):
    gpu_batch_size: Dict = {}
    tensor_path: str = ''

    # Optional fields:
    active_sms: int = 0
    bert_opt_seqlen: int = 0
    cache_file: str = ''
    engine_dir: str = ''
    graph_specs: str = ''
    graphs_max_seqlen: int = 0
    single_stream_expected_latency_ns: int = 0
    single_stream_target_latency_percentile: float = 0.0
    vboost_slider: int = 0
    workspace_size: int = 0


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class C245M8_H100NVL_94GBX2_HighAccuracy(C245M8_H100NVL_94GBX2):
    precision = "fp16"
    single_stream_expected_latency_ns = C245M8_H100NVL_94GBX2.single_stream_expected_latency_ns * 2


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class C245M8_H100NVL_94GBX2_Triton(C245M8_H100NVL_94GBX2):
    use_triton = True

    # Applicable fields for this benchmark are listed below. Not all of these are necessary, and some may be defined in the BaseConfig already and inherited.
    # Please see NVIDIA's submission config files for example values and which fields to keep.
    # Required fields (Must be set or inherited to run):
    gpu_batch_size: Dict = {}
    tensor_path: str = ''

    # Optional fields:
    active_sms: int = 0
    batch_triton_requests: bool = False
    bert_opt_seqlen: int = 0
    cache_file: str = ''
    engine_dir: str = ''
    gather_kernel_buffer_threshold: int = 0
    graph_specs: str = ''
    graphs_max_seqlen: int = 0
    max_queue_delay_usec: int = 0
    num_concurrent_batchers: int = 0
    num_concurrent_issuers: int = 0
    output_pinned_memory: bool = False
    single_stream_expected_latency_ns: int = 0
    single_stream_target_latency_percentile: float = 0.0
    triton_grpc_ports: str = ''
    triton_num_clients_per_frontend: int = 0
    triton_num_frontends_per_model: int = 0
    triton_num_servers: int = 0
    triton_skip_server_spawn: bool = False
    triton_verbose_frontend: bool = False
    use_concurrent_harness: bool = False
    vboost_slider: int = 0
    workspace_size: int = 0


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class C245M8_H100NVL_94GBX2_HighAccuracy_Triton(C245M8_H100NVL_94GBX2_HighAccuracy):
    use_triton = True


