import subprocess

def get_start_cores():
    start_cores = subprocess.check_output('lscpu | grep "NUMA node.* CPU.*" | awk "{print \$4}" | cut -d "-" -f 1', shell=True)
    start_cores = start_cores.decode('ascii').rstrip().split('\n')
    start_cores = [int(_) for _ in start_cores]
    return start_cores