from argparse import ArgumentParser
import os
import sys
import time
import multiprocessing as mp
import array
import numpy as np
import threading
import subprocess
import logging
import collections
import thread_binder
#from profiles import *
import mlperf_loadgen as lg
from numa import memory  

sys.path.insert(0, os.path.join(os.getcwd(),"common"))
from configParser import parseWorkloadConfig

#import intel_pytorch_extension as ipex
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("INTEL-Inference")


MS=1000
SCENARIO_MAP = {
    "singlestream": lg.TestScenario.SingleStream,
    "offline": lg.TestScenario.Offline,
    "server": lg.TestScenario.Server,
}

def get_args():
    
    parser = ArgumentParser("Parses global and workload-specific arguments")
    parser.add_argument("--workload-name", help="Name of workload", required=True)
    parser.add_argument("--scenario", choices=["Offline", "Server"], help="MLPerf scenario to run", default="Offline")
    parser.add_argument("--mlperf-conf", help="Path to mlperf.conf file")
    parser.add_argument("--user-conf", help="Path to user.conf file containing overridden workload params")
    parser.add_argument("--mode", choices=["Accuracy", "Performance"], help="MLPerf mode to run", default="Performance")
    parser.add_argument("--workload-config", help="A json file that contains Workload related arguments for creating sut and dataset instances")
    parser.add_argument("--num-instance", type=int, help="Number of instances/consumers", default=2)
    parser.add_argument("--cpus-per-instance", type=int, help="Number of cores per instance", default=8)
    parser.add_argument("--warmup", type=int, help="Number of warmup iterations", default=10)
    parser.add_argument("--precision", choices=["int8", "bf16", "fp32", "mix"], help="Model precision to run", default="int8")
    parser.add_argument("--workers-per-instance", type=int, help="Number of workers per each instance/consumer", default = 1)
    parser.add_argument("--cores-offset", type=int, help="Cpus to offset on 1st socket", default=0)
    args = parser.parse_args()
    return args


class Consumer(mp.Process):
    def __init__(self, task_queue, out_queue, sut_params, dataset_params, lock, cond_var, init_counter, proc_idx,
                       start_core_idx, mem_idx ,num_cores, args, num_workers=1):
        mp.Process.__init__(self)
        self.num_workers = num_workers
        self.task_queue = task_queue
        self.out_queue = out_queue
        self.lock = lock
        self.cv = cond_var
        self.init_counter = init_counter
        self.proc_idx = proc_idx
        self.args = args
        self.affinity = list(range(start_core_idx, start_core_idx + num_cores))
        self.start_core_idx = start_core_idx
        self.end_core_idx = start_core_idx + num_cores - 1
        self.mem_idx = mem_idx
        self.dataset_params = dataset_params

        self.num_cores = num_cores
        self.cpus_per_worker = num_cores // num_workers
        self.workers = []
        self.sut_params = sut_params
        self.out_queue = out_queue
        self.warmup_count = args.warmup
        self.latencies = collections.defaultdict(list)
        log.info("Cores: {}-{}".format(self.start_core_idx, self.end_core_idx))


    def do_warmup(self):
        warmup_data = self.data_obj.get_warmup_samples()
        log.info("Starting warmup with {} samples".format(self.warmup_count))
        for idx in range(self.warmup_count):
            log.info("idx {}".format(idx))
            output = self.sut_obj.predict(warmup_data.data)
        log.info("Warmup Completed")
        with self.cv:
            self.init_counter.value += 1
            self.cv.notify()


    def handle_tasks(self, i, task_queue, result_queue, args, pid, start_core, end_core):
        thread_binder.bind_thread(start_core, end_core - start_core + 1)
        pid = os.getpid()
        worker_name = str(pid) + "-" + str(i)
        os.environ['OMP_NUM_THREADS'] = '{}'.format(end_core - start_core + 1)

        #self.lock.acquire()
        #self.init_counter.value += 1
        #self.lock.release()

        if (self.warmup_count>0):
            self.do_warmup()
        else:
            with self.cv:
                self.init_counter.value += 1
                self.cv.notify()


        log.info("Start worker on {} - {}".format(start_core, end_core))

        #TODO: Enable profiling here
        while True:
            next_task = task_queue.get()
            if next_task is None:
                log.info("{} : Exiting ".format(worker_name))
                break

            query_id_list = next_task.query_id_list
            sample_index_list = next_task.sample_index_list

            data = self.data_obj.get_samples(sample_index_list)

            output = self.sut_obj.predict(data.data)

            result = self.data_obj.post_process(query_id_list, sample_index_list, output)

            result_queue.put(result)
            task_queue.task_done()


    def run(self):
        self.proc_idx = self.pid
        # os.environ['OMP_NUM_THREADS'] = '{}'.format(self.end_core_idx-self.start_core_idx+1)
        os.environ['OMP_NUM_THREADS'] = '4'
        os.sched_setaffinity(0, self.affinity)
        memory.set_membind_nodes(self.mem_idx)

        # Load model
        log.info("Loading model")
        from Backend import Backend
        self.sut_obj = Backend(**self.sut_params)
        model = self.sut_obj.load_model()

        # Load dataset (if not loaded already)
        from Dataset import Dataset
        self.data_obj = Dataset(**self.dataset_params)

        self.data_obj.load_dataset()
        log.info("Available samples: {} ".format(self.data_obj.count))

        cur_step = 0
        log.info("Start testing...")

        if self.num_workers >= 1:
            start_core = self.start_core_idx
            cores_left = self.num_cores

            for i in range(self.num_workers):
                end_core = start_core + self.cpus_per_worker - 1
                cores_left -= self.cpus_per_worker
                
                #TODO: Move this to workload config. Remove hardcoded constraints
                if cores_left < 2:
                    end_cores = self.end_core_idx
                worker = mp.Process(target=self.handle_tasks, args=(i, self.task_queue, self.out_queue, self.args, self.pid, start_core, end_core))

                self.workers.append(worker)
                #worker.start()
                start_core += self.cpus_per_worker

                if cores_left < 2:
                    break

            for w in self.workers:
                w.start()

            for w in self.workers:
                w.join()
            log.info("{} : Exiting consumer process".format(os.getpid()))
        else:
            self.handle_tasks(0, self.task_queue, self.out_queue, self.args, self.pid, self.start_core_idx, self.end_core_idx)


def response_loadgen(out_queue):

    while True:
        next_task = out_queue.get()
        if next_task is None:
            # None means shutdown
            log.info('Exiting response thread')
            break
        query_id_list = next_task.query_id_list
        result = next_task.result
        array_type_code = next_task.array_type_code

        batch_size = len(query_id_list)

        for id, out in zip(query_id_list, result):
            response_array = array.array(array_type_code, out)
            bi = response_array.buffer_info()
            responses = [lg.QuerySampleResponse(id, bi[0], bi[1]*response_array.itemsize)]
            lg.QuerySamplesComplete(responses)

def flush_queries():
    pass

#def process_latencies(latencies):
#    pass

def load_query_samples(query_samples):
    pass

def unload_query_samples(query_samples):
    pass

def get_physical_start_cores_idx_per_numa_node(num_cpus):
    numa_nodes_path = "/sys/devices/system/node"
    nodes = [d for d in os.listdir(numa_nodes_path) if d.startswith("node")]

    node_cores = {}

    for node in nodes:
        cpulist_path = os.path.join(numa_nodes_path, node, "cpulist")
        try:
            with open(cpulist_path) as f:
                cpus = f.read().strip()
                
                # Split by commas or dashes to handle ranges of cores
                first_core = cpus.split(",")[0]  # Extract the first set of cores
                node_number = node.replace("node", "")  # Remove the word "node"
                node_cores[node_number] = first_core
        except FileNotFoundError:
            print(f"File not found for {node}")

    node_start_idx_map = {}
    for node, first_core in node_cores.items():
        # print(f"{node}: {first_core}")
        start, end = map(int, first_core.split('-'))
        # print(list(range(start, end + 1)))
        cores = (list(range(start, end + 1)))
        i = 0
        core_start = []
        for core in cores:
            if core == cores[0]:
                core_start.append(core)
            i+=num_cpus
            if core_start and i<=(len(cores)-num_cpus):
                core_start.append(cores[i])
        node_start_idx_map[node] =  core_start 


    return node_start_idx_map


def main():
    args = get_args()

    # Get workload config parameters
    backend_params, dataset_params, enqueue_params, buckets, num_resp_qs, import_path = parseWorkloadConfig(args.workload_config)

    sys.path.insert(0, os.path.join(os.getcwd(),import_path))

    # Imports
    from InQueue import InQueue

    global num_cpus
    
    log.info(args)
    scenario = args.scenario
    mode = args.mode

    # TODO: Need to validate the cpu-instance combo is valid on the system
    # num_ins = args.num_instance
    num_cpus = args.cpus_per_instance
    ins_per_consumer = args.workers_per_instance

    # Establish communication queues
    lock = mp.Lock()
    init_counter = mp.Value("i", 0)
    cv = mp.Condition(lock=lock)

    manager = mp.Manager()

    settings = lg.TestSettings()
    settings.scenario = SCENARIO_MAP[scenario.lower()]
    #DEPRECATED v5.0: settings.FromConfig(args.mlperf_conf, args.workload_name, scenario)
    settings.FromConfig(args.user_conf, args.workload_name, scenario)
    settings.mode = lg.TestMode.AccuracyOnly if mode.lower()=="accuracy" else lg.TestMode.PerformanceOnly

    consumers = []
    loadgen_cores = args.cores_offset
    thread_binder.bind_thread(0, 2)

    # TODO: Assign response queues to instances based on socket/numa config
    out_queue = [mp.Queue() for _ in range(num_resp_qs)]
    cores_idx_per_node = get_physical_start_cores_idx_per_numa_node(num_cpus)
    start_core_idx_list = []
    start_memory_idx_list = []
    for node , cores in cores_idx_per_node.items():
        for core in cores:
            start_core_idx_list.append(core)
            start_memory_idx_list.append(node)
    if len(buckets)==0:
        # If no 'bucketing', all consumers/instances fetch from a single input queue
        input_queues = mp.JoinableQueue()

        total_workers = len(start_core_idx_list) * ins_per_consumer 

        i = 0
        # TODO: Consider system config i.e core-socket-numa allocation
        while i < len(start_core_idx_list):
            # start_core_idx = i * num_cpus + loadgen_cores
            start_core_idx = start_core_idx_list[i]
            mem_idx = start_memory_idx_list[i]
            consumer = Consumer(input_queues, out_queue[i%num_resp_qs], backend_params, dataset_params, lock, cv, init_counter, i, start_core_idx,mem_idx, num_cpus, args, num_workers=ins_per_consumer)
            consumers.append(consumer)
            i += 1
    else:
        batch_sizes = buckets["batch_sizes"]
        cutoffs = buckets["cutoffs"]
        bucket_instances = buckets["instances"]
        cpus_per_instances = buckets["cores_per_bucket_instances"]
        ins_per_bucket_consumers = buckets.get("instances_per_bucket_consumers",[1]*len(cutoffs))

        input_queues = [mp.JoinableQueue() for _ in range(len(cutoffs))]
        total_workers = 0
        i = 0
        # start_core_idx = 0 + loadgen_cores
        for j, cutoff in enumerate(cutoffs):
            batch_size = batch_sizes[j]
            num_cpus = cpus_per_instances[j]
            num_ins = bucket_instances[j]
            ins_per_consumer = ins_per_bucket_consumers[j]
            total_workers += num_ins
            
            for k in range(num_ins):
                # TODO: Need to align with NUMA configuration 
                #       so that each consumer's cores are on same NUMA node
                log.info("Assigning to output queue {}".format( i % num_resp_qs))
                start_core_idx = start_core_idx_list[k]
                mem_idx = start_memory_idx_list[k]
                consumer = Consumer(input_queues[j], out_queue[i%num_resp_qs], backend_params, dataset_params, lock, cv, init_counter, i, start_core_idx, mem_idx,num_cpus, args, num_workers=ins_per_consumer)
                consumers.append(consumer)
                
                start_core_idx += num_cpus
                i += 1
    
    # Update enqueue parameters and instantiate object        
    enqueue_params.update(mpQueue=input_queues, **buckets) #, qsl=datasetObj)
    enqueueObj = InQueue(**enqueue_params)

    for c in consumers:
        c.start()

    # Wait until all sub-processors are ready
    #while init_counter.value < total_workers:
    #    time.sleep(2)
    with cv:
        cv.wait_for(lambda : init_counter.value==total_workers)

    log.info("init_count {} total_workers {}\n".format(init_counter.value, total_workers))
    # Start response thread(s)
    resp_workers = [threading.Thread(
        target=response_loadgen, args=(out_queue[i],)) for i in range(num_resp_qs)]

    for resp_worker in resp_workers:
        resp_worker.daemon = True
        resp_worker.start()

    def issue_queries(query_samples):
        enqueueObj.put(query_samples, receipt_time=time.time())


    sut = lg.ConstructSUT(
#        issue_queries, flush_queries, process_latencies)
        issue_queries, flush_queries)
    qsl = lg.ConstructQSL(
        dataset_params['total_sample_count'], min(dataset_params['total_sample_count'], settings.performance_sample_count_override), load_query_samples, unload_query_samples)

    log_path = "output_logs"
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    log_output_settings = lg.LogOutputSettings()
    log_output_settings.outdir = log_path
    log_output_settings.copy_summary_to_stdout = True
    log_settings = lg.LogSettings()
    log_settings.log_output = log_output_settings

    lg.StartTestWithLogSettings(sut, qsl, settings, log_settings)
    log.info("Test completed")

    if len(buckets) > 0:
        for q in input_queues:
            for i in range(bucket_instances[j]):
                q.put(None)
    else:
        for _ in range(init_counter.value):
            input_queues.put(None)

    for c in consumers:
        c.join()
    
    for out_q in out_queue:
        out_q.put(None)

    lg.DestroyQSL(qsl)
    lg.DestroySUT(sut)


    
if __name__ == "__main__":
    main()
