from . import *


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class ARS_111GL_NHR(ServerGPUBaseConfig):
    system = KnownSystem.ARS_111GL_NHR
    gpu_batch_size = {'dlrm-v2': 204800}
    embedding_weights_on_gpu_part: float = 1.0
    server_target_qps = 81000 


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class ARS_111GL_NHR_High_Accuracy(ARS_111GL_NHR):
    server_target_qps = 50480
    interaction_op_precision = 'fp16'
