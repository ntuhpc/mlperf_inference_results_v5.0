import subprocess
import os
import utils
import math
import time

cores_per_inst = int(os.environ.get("CORES_PER_INST", "1"))
num_numa_nodes = int(os.environ.get("NUM_NUMA_NODES", "1"))
nodes_per_inst = int(os.environ["NUM_NUMA_NODES"])/int(os.environ["NUM_INSTS"])
insts_per_node = int(os.environ["INSTS_PER_NODE"])
tp_size = int(os.environ["TP_SIZE"])

if nodes_per_inst <= 1:
   insts_per_node = int(os.environ["INSTS_PER_NODE"])
else:
   insts_per_node = 0

def main():
    node_start_cores = utils.get_start_cores()
    core_lists = []
    if insts_per_node>0:
        for i in range(num_numa_nodes):
            for j in range(insts_per_node):
                core_lists.append(list(range(node_start_cores[i]+j*cores_per_inst, node_start_cores[i]+(j+1)*cores_per_inst)))
    
    num_workers = (insts_per_node*num_numa_nodes + tp_size-1)//tp_size

    processes = []
    for j in range(num_workers):
        tp_size_c = min(num_numa_nodes*insts_per_node - j*tp_size, tp_size)
        print(j, tp_size_c)
        for i in range(1, tp_size_c):
            node = math.floor(nodes_per_inst*(j*tp_size+i))
            core_list = tuple(core_lists[j*tp_size+i])
            if i==1:
                head_cmd = f"--head --port {6379+j} "
            else:
                head_cmd = f"--address localhost:{6379+j} "
            cmd = f"numactl -C {core_list[0]}-{core_list[-1]} -m {node} ray start {head_cmd}--num-cpus={cores_per_inst} --num-gpus=0"
            print(cmd)
            processes.append(subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE))
            time.sleep(4)

    for process in processes:
        process.wait()


if __name__ == "__main__":
    main()
