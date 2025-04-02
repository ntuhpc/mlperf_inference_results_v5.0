# Bandwidth specs of datacenter systems for MLPerf Inference 5.0

As per the [MLPerf Datacenter Bandwidth Requirements](https://github.com/mlcommons/inference_policies/blob/master/inference_rules.adoc#appendix-b-datacenter-bandwidth-requirements) , Datacenter systems must satisfy both the ingress and egress bandwidth requirements for each benchmark.

Intel submitted using the following datacenter systems for respective workload submission:

|System                 |Workloads submitted                                |
|-----------------------|---------------------------------------------------|
|INTEL(R) XEON(R) 6980P | Resnet50-v1.5,Retinanet,3D UNET,GPT-J,DLRMv2,RGAT |


## Network bandwidth of Intel's systems

### INTEL(R) XEON(R) 6980P
Memory configuration : 24 slots per socket / 96GB each / 8800 MT/s MCR
MCR memory transfers 64 bits (8 bytes) per transfer
Total Bandwidth: 8800 MT/s × 8 bytes(MCR memory transfers 64 bits (8 bytes) per transfer) ×24(slots) = 1.6896 TB/s

## Max permissible QPS per system

Below table shows max permissible throughput per workload, calculated using the system bandwidth. Througput numbers below are well above the submission throughput.


| System  | Bandwidth of system (bytes/sec)| ResNet50 | DLRM  | 3D U-Net | GPT-J  |RetinaNet |Llama2-70B | Llama2-70B-Interactive  | Llama3-405B |
|---------|--------------------------------|----------|-------|----------|--------|----------|-----------|-------------------------|-------------|
| INTEL(R) XEON(R) 6980P |1,689.6× 10^9    | 11.3*10^6| 4.8*10^7  | 51285| 22*10^7| 8.8*10^5 |    NA     |          NA             |    NA       |

