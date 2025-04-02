# Bandwidth measurements of Trillium to network for MLPerf Inference 5.0

## Introduction
As of MLPerf Inf v4.0, submissions are required to prove that the systems used provide a certain level of ingress (network to device) and egress (device to network) bandwidth bypassing the CPU.

This round, Google is making submissions using the following systems:
- Trillium (v6e4)

## Procedure
This document assumes that the following architecture is used in all systems. Each device(TPU) is connected to the PCIe bus. The PCIe bus is also connected to each NIC (network interface cards). Hence, the ingress data flow is: 
1. network to NIC
2. NIC to PCIe bus
3. PCIe bus to device (TPU)

The egress data flow is the reverse:
1. Device to PCIe
2. PCIe to NIC
3. NIC to network

Hence, we must measure (1) Device to PCIe bandwidth and (2) PCIe to NIC bandiwdth. The minimum of (1) and (2) is the theoretical max bandwidth that our system will support.  

### Getting PCIe to NIC bandwidth
First, we get the number of NICs that are available for use, using `networkctl` and `lshw`. Below is the information from L40S.

```
$ networkctl
IDX LINK        TYPE     OPERATIONAL SETUP     
  1 lo          loopback carrier     unmanaged
  2 ens8        ether    routable    configured
  3 docker0     bridge   routable    unmanaged
  5 vethf9c995f ether    degraded    unmanaged


$ sudo lshw -class network -businfo
Bus info          Device           Class          Description
=============================================================
pci@0000:00:08.0  ens8             network        Compute Engine Virtual Ethernet [gVNIC]

```

`lshw` tells us that we use one of the NICs 

1 gVNIC corresponds to 1 pNIC,  2 gVNICs corresponds to 2 pNICs and 3 gVNICs will still correspond to 2 pNICs because we are using Trillium - x4 which is half of a single host.

1 pNIC is equivalent to 200 Gbps [https://cloud.google.com/tpu/docs/v6e#system-architecture]


### Theoretical maximum bandwidth of system
Assuming the system is bound by VM capabilities, the theoretical bandwidth of the system is bound by 1 pNIC which is 200Gbps for 4 devices. We can increase it further by 1 more pNIC by creating another gVNIC which will add 200 Gbps on top.

## Bandwidth requirements per scenario
For each benchmark model, the offline scenario uses the most bandwidth. We list the bandwidth used per model `used_bw` (in byte/sec) for a run in Offline scenario.  
The `used_bw` is a function of throughput of input samples `tput` (sample/second) and size of each input sample `bytes/sample`.  

### Ingress bandwidth requirements

| Benchmark | Formula                                                                                                               | Bandwidth used                  | Values                                                                                                               |
|-----------|-----------------------------------------------------------------------------------------------------------------------|---------------------------------|----------------------------------------------------------------------------------------------------------------------| 
| SDXL      | ```used_bw = num_inputs x max_prompt_len x dtype_size```                                                              | ```used_bw = tput x 308```      | ```num_inputs = 1; max_prompt_len = 77```                                                                            |

### Egress bandwidth requirements

According to the [rules set out by MLCommons](https://github.com/mlcommons/inference_policies/blob/master/inference_rules.adoc#b2-egress-bandwidth), we only need to measure egress bandwidth for 3D U-Net and SDXL. 
| Benchmark | Formula                                                                                                               | Bandwidth used                  | Values                                                                                                               |
|-----------|-----------------------------------------------------------------------------------------------------------------------|---------------------------------|----------------------------------------------------------------------------------------------------------------------|
| SDXL      | ```used_bw = num_inputs x image_height x image_width x image_channel x dtype_size```                                  | ```used_bw = tput x 3145728```  | ```num_inputs = 1; image_height = image_width = 1024; image_channel = 3```                                           |

## Calculating maximum permissible QPS per system-model submission pair
Using the formulae in the previous section, we calculate for each system-benchmark pair the maximum permissible QPS by setting `system_bw = used_bw` and calculating for `tput`.
For workloads with constraint on both ingress and egress bandwidth, we take the max of the two. (for example, we take egress for SDXL)

PLEASE NOTE - The numbers are calculated below for Google's  systems and are provided for reference only. Each systems configuration (and hence, bandwidth) may be different. It is imperative that each participant does such calculations individually for their own systems.

| System                       | Bandwidth of system | SDXL         |
|------------------------------|---------------------|-------------|
|Trillium v6e-4                     | 50GB/s             | 17066 (50GB/3145728B) |