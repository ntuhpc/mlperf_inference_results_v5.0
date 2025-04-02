from . import *

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class ARS_111GL_NHR(OfflineGPUBaseConfig):
    system = KnownSystem.ARS_111GL_NHR
    gpu_batch_size = {'dlrm-v2': 600_000}
    embedding_weights_on_gpu_part: float = 1.0
    offline_expected_qps = 87000


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class ARS_111GL_NHR_High_Accuracy(ARS_111GL_NHR):
    offline_expected_qps = 53000
    interaction_op_precision = 'fp16'


