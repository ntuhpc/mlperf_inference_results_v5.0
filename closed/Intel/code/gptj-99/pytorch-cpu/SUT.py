import os
import time
import numpy as np
import math
import array
import torch
import intel_extension_for_pytorch as ipex
from torch.nn.functional import pad
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, MixtralForCausalLM, LogitsProcessor, LogitsProcessorList
from transformers.generation.streamers import BaseStreamer

import pickle
import time
import threading
import torch.multiprocessing as mp
import queue

import logging
from typing import TYPE_CHECKING, Optional, List
from pathlib import Path

import mlperf_loadgen as lg
from dataset import Dataset
from dataset_cnn import Dataset_CNN
from numa import schedule, memory
import os
import ray
import utils
from tqdm import tqdm

from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("SUT")

gen_kwargs = {
    "early_stopping": True,
    "max_new_tokens": 1024,
    "min_new_tokens": 1,
    "num_beams": 1,
    "do_sample": False
}

cores_per_inst = int(os.environ.get("CORES_PER_INST", "1"))
num_numa_nodes = int(os.environ.get("NUM_NUMA_NODES", "1"))
nodes_per_inst = int(os.environ["NUM_NUMA_NODES"])/int(os.environ["NUM_INSTS"])
insts_per_node = int(os.environ["INSTS_PER_NODE"])

if nodes_per_inst <= 1:
   insts_per_node = int(os.environ["INSTS_PER_NODE"])
else:
   insts_per_node = 0

tokens_between = int(os.environ.get("TOKENS_BETWEEN", "1"))

def void(*args, **kwargs):
    pass

pbar = int(os.environ.get("PBAR", "1"))
if pbar==1:
    print = void

class StopAfterSequence(LogitsProcessor):
        """Logits processor (to use with HuggingFace `generate()` method :
        https://huggingface.co/docs/transformers/v4.24.0/en/main_classes/
        text_generation#transformers.generation_utils.GenerationMixin).

        This logits processor makes that when the model generates a specified
        stopping sequence, it stops generating new tokens

        Args:
            stop_seq (List[int]): ID of the space token.
            eos_token_id (int): ID of the EOS token.
            device (str): Device that the model is running
        """
        def __init__(self, eos_token_id: int, stop_seq: List[int] = [13, 13940, 28832, 13], device="cpu"):
            super().__init__()
            assert(len(stop_seq) >= 1)
            self.device = device
            self.stop_seq = stop_seq
            self.stop_seq_length = len(stop_seq)
            self.eos_token_id = eos_token_id

        def check_stop_condition(self, input_ids: torch.LongTensor):
            stop_condition_met = (input_ids[-self.stop_seq_length:] == self.stop_seq)#.all()
            return stop_condition_met
        
        def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
            
            if len(input_ids) > self.stop_seq_length:
                forced_eos = torch.full((scores.size(0),), -float("inf")).to(self.device)
                forced_eos[self.eos_token_id] = 0
                if self.check_stop_condition(input_ids):
                    scores = forced_eos
            return scores


class Instance(mp.Process):
    def __init__(
        self,
        model_path=None,
        workload_name="mixtral",
        dataset_path=None,
        device="cpu",
        batch_size=-1,
        total_sample_count=-1,
        rank=-1,
        dtype=torch.bfloat16,
        core_list=(),
        node_list=(),
        input_queue=None,
        output_queue=None,
        first_queue=None,
        cond_var=None,
        alive_counter=None,
        dead_counter=None,
        sample_counter=None,
        current_counter=None,
        server=False,
        tp=1,
        quantized=False,
        warmup=False
    ):
        mp.Process.__init__(self)
        self.model_path = model_path
        self.workload_name = workload_name
        self.dataset_path = dataset_path
        self.device=device
        self.batch_size=batch_size
        self.total_sample_count=total_sample_count
        self.rank = rank
        self.dtype = dtype
        self.core_list = core_list
        self.node_list = node_list
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.first_queue = first_queue
        self.cond_var = cond_var
        self.alive_counter = alive_counter
        self.dead_counter = dead_counter
        self.sample_counter = sample_counter
        self.current_counter = current_counter
        self.query_idx_mapping = []
        self.qid_mapping = []
        self.start_time_mapping = []
        self.wait_time_mapping = []
        self.first_time_mapping = []
        self.finished = False
        self.server = server
        self.tp = tp
        self.quantized = quantized
        self.wait_for_next_sample = 0
        self.warmup = warmup
        self.tpot_list = []
        self.tprefill_list = []
        self.nprefill_list = []

        # self.time_last_prefill = 0

    def run(self):
        if self.tp>1:
            os.environ["RAY_ADDRESS"]=f"localhost:{6379+self.rank}"
            node_list = tuple([math.floor(node) for node in self.node_list])
            memory.set_membind_nodes(*node_list)
            schedule.run_on_cpus(os.getpid(), *self.core_list)
            print(f"Binding rank {self.rank} to nodes {node_list}")
            print(f"Binding rank {self.rank} to cores {self.core_list}")
            
            with self.cond_var:
                self.dead_counter.value += 1
                self.cond_var.notify()
            self.load_model()
            
        else:
            node_list = tuple([math.floor(node) for node in self.node_list])
            memory.set_membind_nodes(*node_list)
            schedule.run_on_cpus(os.getpid(), *self.core_list)
            print(f"Binding rank {self.rank} to nodes {node_list}")
            print(f"Binding rank {self.rank} to cores {self.core_list}")

            with self.cond_var:
                self.dead_counter.value += 1
                self.cond_var.notify()
            self.load_model()

        self.load_dataset()

        self.warmup_req_idx = set([])
        if self.warmup:
            print(f"Doing warmup for rank {self.rank}")
            self.do_warmup()
        else:
            with self.cond_var:
                self.alive_counter.value += 1
                self.cond_var.notify()

        keep_alive = True
        self.req_counter = 0 # Request counter

        while keep_alive:
            keep_alive = self.process_queries()

        with self.cond_var:
            self.alive_counter.value -= 1
            # The instance needs to get restarted
            if self.alive_counter.value == 0:
                # self.output_queue.put(None)
                pass
            else:
                self.input_queue.put(None)
            self.cond_var.notify()
        
    def load_dataset(self):
        if self.workload_name=="gptj":
            self.data_object = Dataset_CNN(self.dataset_path,
                                        model_checkpoint_path=self.model_path,
                                        total_sample_count=self.total_sample_count,
                                        pad_inputs=False)
            self.data_object.loadDataset()

        elif "mixtral" in self.workload_name:
            self.data_object = Dataset(self.model_path,
                                            dataset_path=self.dataset_path,
                                            total_sample_count=self.total_sample_count,
                                            device=self.device)
        else:
            self.data_object = Dataset(self.model_path,
                                        dataset_path=self.dataset_path,
                                        total_sample_count=self.total_sample_count,
                                        device=self.device)

    def load_model(self):
        self.model = LLM(
            model=self.model_path,
            dtype='bfloat16',
            skip_tokenizer_init=False,
            tensor_parallel_size=self.tp,
            max_num_seqs=512,
            block_size=32
            )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            model_max_length=1024,
            padding_side="left",
            use_fast=False
        )

        if self.workload_name=="gptj": # gpt-j
            self.sampling_params = SamplingParams(
                max_tokens=128,
                min_tokens=30,
                temperature=0.0,
                detokenize=False,
                use_beam_search=True,
                early_stopping=True,
                best_of=4
            )
        else: # MoE
            self.sampling_params = SamplingParams(
                max_tokens=1024,
                min_tokens=1,
                temperature=0.0,
                detokenize=False
            )
            logits_processor = LogitsProcessorList([StopAfterSequence(self.tokenizer.eos_token_id)])
            self.sampling_params_mbxp = SamplingParams(
                max_tokens=1024,
                min_tokens=1,
                temperature=0.0,
                detokenize=False,
                logits_processors=logits_processor
            )
            print("Loaded model")

        self.tokenizer.pad_token = self.tokenizer.eos_token
        print("Loaded tokenizer")

    def do_warmup(self):

        first_token_id = []
        #self.warmup_req_idx = set([]) # Defined in self.run
        start_time = []
        first_time = dict()
        input_prompts = []
        max_tokens = self.sampling_params.max_tokens
        min_tokens = self.sampling_params.min_tokens
        sampling_params = self.sampling_params

        warmup_sample_idxs = torch.randint(0, len(self.data_object.input_lens), (1,))
        
        for j in range(1):
            if self.workload_name=="gptj":
                input_ids,_,_ = self.data_object.getSamples([warmup_sample_idxs[j].item()])
            else:
                input_ids = self.data_object.input_ids[warmup_sample_idxs[j].item()]
            inputs = TokensPrompt(prompt_token_ids=input_ids[0].tolist())
            input_prompts.append(inputs)

        ids = torch.randint(100000, 200000, (1,))
        ids_map = dict()
        for j, prompt in enumerate(input_prompts):
            id = ids[j].item()
            ids_map[id] = j
            self.warmup_req_idx.add(id)
            start_time.append(time.time())
            
            self.model.llm_engine.add_request(str(id), prompt, self.sampling_params)
            outputs = self.model.llm_engine.step()
            
            for output in outputs:
                req_id = int(output.request_id)
                if output.finished:
                    token_ids = output.outputs[0].token_ids
                    finish_time = time.time()
                    time_elapsed = finish_time - start_time[ids_map[req_id]]
                    ttft = first_time[req_id]
                    tpot = (time_elapsed - ttft) / (len(token_ids) - 1)
                    print(f"Rank {self.rank}: Query {ids_map[req_id]} finished after {time_elapsed:.2f}s, "
                          f"TTFT {ttft:.2f}s, "
                          f"TPOT {tpot:.2f}s, "
                          f"generated {len(token_ids)} tokens")
                    

                if req_id not in first_time:
                    index = ids_map[req_id]
                    diff_time = time.time() - start_time[index]
                    first_time[req_id] = diff_time

        while self.model.llm_engine.has_unfinished_requests():
            outputs = self.model.llm_engine.step()
            for output in outputs:
                req_id = int(output.request_id)
                if output.finished:
                    token_ids = output.outputs[0].token_ids
                    finish_time = time.time()
                    time_elapsed = finish_time - start_time[ids_map[req_id]]
                    ttft = first_time[req_id]
                    tpot = (time_elapsed - ttft) / (len(token_ids) - 1)
                    print(f"Rank {self.rank}: Query {ids_map[req_id]} finished after {time_elapsed:.2f}s, "
                          f"TTFT {ttft:.2f}s, "
                          f"TPOT {tpot:.2f}s, "
                          f"generated {len(token_ids)} tokens")

                if req_id not in first_time:
                    diff_time = time.time() - start_time[ids_map[req_id]]
                    first_time[req_id] = diff_time

        with self.cond_var:
            print(f"Worker {self.rank} warmup done")
            self.alive_counter.value += 1
            print(f"Alive counter value {self.alive_counter.value}")
            self.cond_var.notify()


    def process_queries(self):
        self.wait_for_next_sample -= 1
        self.wait_for_next_sample = max(0,self.wait_for_next_sample)
        samples_to_fill = self.batch_size - self.model.llm_engine.get_num_unfinished_requests()

        # Not expecting server batch size to exceed 64
        if self.batch_size<=64:
            samples_to_fill = min(1,samples_to_fill)
        add_new_qitem = (self.wait_for_next_sample==0) and (samples_to_fill>0)
        return_value = True
        qitem = -1
        
        if self.finished:
            add_new_qitem = False
            if not self.model.llm_engine.has_unfinished_requests():
                return_value = False

        qitem_list = []
        while add_new_qitem and len(qitem_list)<samples_to_fill:
            try:
                qitem = self.input_queue.get(False)
            except:
                qitem = -1
            else:
                pass
            
            if qitem is None:
                self.finished = True
                add_new_qitem = False
            elif type(qitem) == int:
                add_new_qitem = False
            else:
                qitem_list.append(qitem)

        tik1 = time.time()
        tik2 = time.time()
        first_token_id = []

        if len(qitem_list)>0:
            # Block instance when doing prefill
            with self.cond_var:
                self.current_counter.value += 999
                self.cond_var.notify()

        for qitem in qitem_list:
            qitem, start_time = qitem
            query_ids = [q.index for q in qitem]
            qid = [q.id for q in qitem]
            # Construct / collate batch

            input_ids_tensor = []
            # q = qitem[0] # All qitems are of length 1
            for i,q in enumerate(qitem):
                sampling_params = self.sampling_params
                if self.workload_name=="gptj":
                    input_ids_tensor,_,_ = self.data_object.getSamples([q.index])
                else:
                    input_ids_tensor = self.data_object.input_ids[q.index]
                    if q.index>=10000:
                        sampling_params = self.sampling_params_mbxp

                first_token_id.append(len(self.query_idx_mapping))
                self.wait_time_mapping.append(time.time()-start_time[i])
                self.query_idx_mapping.append(q.index)
                self.qid_mapping.append(q.id)
                self.start_time_mapping.append(time.time())
                self.first_time_mapping.append(0)
                self.tprefill_list.append(0)
                self.nprefill_list.append(0)

                requests_data = []
                for prompt_id in input_ids_tensor:
                    inputs = TokensPrompt(prompt_token_ids=prompt_id.tolist())
                    requests_data.append({
                        "inputs": inputs,
                        "params": sampling_params,
                    })

                for request_data in requests_data:
                    self.model.llm_engine.add_request(str(self.req_counter),
                            request_data["inputs"],
                            request_data["params"])

                    self.req_counter += 1

        if len(qitem_list)>0:
            self.wait_for_next_sample += tokens_between

        pred_output_tokens = []
        query_ids = []
        qid = []
        pred_first_tokens = []
        query_ids_first = []
        qid_first = []
        if self.model.llm_engine.has_unfinished_requests():
            #print("** Calling engine step **")
            if len(first_token_id)>0:
                end = time.time()
            step_outputs = self.model.llm_engine.step()
            if len(first_token_id)>0:
                step_time = time.time()-end
            #print("*** Engine Step called ***")
            for output in step_outputs:
                request_id = int(output.request_id)

                if request_id in self.warmup_req_idx: # Skip requests that are from the warmup 
                    #print("Request is from warmup. Skip")
                    continue

                if len(first_token_id) > 0:
                    # Add step time to everything other than the one being prefilled
                    for i in range(len(self.tprefill_list)-len(first_token_id)):
                        self.tprefill_list[i] += step_time
                        self.nprefill_list[i] += len(first_token_id)

                if output.finished:
                    token_ids = output.outputs[0].token_ids
                    pred_output_tokens.append(token_ids)
                    query_ids.append(self.query_idx_mapping[request_id])
                    qid.append(self.qid_mapping[request_id])
                    if not pbar:
                        finish_time = time.time()
                        time_elapsed = finish_time - self.start_time_mapping[request_id]
                        ttft = self.first_time_mapping[request_id]
                        # Rest of the tokens
                        tpot = (time_elapsed - ttft)/(len(token_ids)-1)
                        self.tpot_list.append(tpot)
                        tpot_sorted = np.sort(self.tpot_list)
                        tpot_mean = np.mean(self.tpot_list)
                        print(f"Query {request_id} finished after {time_elapsed:.2f}s, "
                            f"Rank {self.rank}, "
                            f"TTFT {ttft:.2f}s, "
                            f"TPOT {tpot:.2f}s, "
                            f"prefill time {self.tprefill_list[request_id]:.2f}s, "
                            f"prefill count {self.nprefill_list[request_id]}, "
                            f"wait time {self.wait_time_mapping[request_id]:.2f}s, "
                            f"bs_c {self.model.llm_engine.get_num_unfinished_requests()}, "
                            f"p99 tpot {tpot_sorted[int(0.99*len(tpot_sorted))]:.3f}s, "
                            f"p50 tpot {tpot_mean:.3f}s, "
                            f"generated {len(token_ids)} tokens")
                
                
                if request_id in first_token_id:
                    # Use first token for server tuning even when first token latency is not needed
                    self.first_time_mapping[request_id] = time.time()-self.start_time_mapping[request_id]
                    if self.server:
                        token_ids = output.outputs[0].token_ids
                        pred_first_tokens.append(token_ids)
                        query_ids_first.append(self.query_idx_mapping[request_id])
                        qid_first.append(self.qid_mapping[request_id])

        if len(qitem_list)>0:
        # if (time.time()-self.time_last_prefill>(2/self.tp)) and (self.current_counter.value>100):
            # Unblock instance after prefill
            with self.cond_var:
                self.current_counter.value -= 999
                self.cond_var.notify()
        
        tik3 = time.time()

        if len(pred_first_tokens)>0:
            processed_first = []
            for pred_first_token in pred_first_tokens:
                processed_first.append(np.array(pred_first_token).reshape(-1))
            self.first_queue.put((qid_first, processed_first))

        if len(pred_output_tokens)>0:
            processed_output = self.data_object.postProcess(pred_output_tokens,
                                                            input_seq_lens=0,
                                                            query_id_list=query_ids)

            self.output_queue.put((qid,processed_output))
            tok = time.time()

            with self.cond_var:
                self.sample_counter.value += len(query_ids)
                print(f"Samples run: {self.sample_counter.value}, rank {self.rank}, current_count {self.current_counter.value}")
                self.current_counter.value -= len(query_ids)
                self.cond_var.notify()
        return return_value


class FirstTokenStreamer(BaseStreamer):
    """ Streams first tokens to a 'holder' """

    def __init__(self, first_token, tokens_cache=[], is_first_token=True, response_ids=[] ):
        """ Response ids added to 'sign' the first token"""

        self.first_token = first_token # Queue for first token
        self.is_first_token = is_first_token

        # Cache for subsequent generated tokens
        self.tokens_cache = tokens_cache

        self.response_ids = response_ids

        self.is_prompt = True # The first tokens sent to the streamer are actually the input prompts

    def put(self, value):
        """ Caches the tokens as they're generated. Assumes bs=1 """

        # Prompts are streamed first so we need to skip the first time value that arrives
        if self.is_prompt:
            self.is_prompt = False
            return

        value = value.item()
        if self.is_first_token:

            # Add generated first token together with its query response_id to first tokens queue
            self.first_token.put((value, self.response_ids[0]))

            self.is_first_token = False
            return

        self.tokens_cache.append(value)


    def end(self):
        pass

    def get_out_tokens(self):
        return self.tokens_cache

def calculate_compute_latency(input_len):
    # # 200 input 0.2, 2000 input 2
    # return 0.001*input_len
    # 0 input 0.2 2000 input 1.8
    # return 0.2 + 0.0008*input_len
    return 0.0008*input_len

class SUT():
    def __init__(self,
                 model_path=None,
                 workload_name="llama2-70b",
                 dtype="bfloat16",
                 device="cpu",
                 batch_size=None,
                 total_sample_count=24576,
                 dataset_path=None,
                 use_cached_outputs=False,  # Set this to True *only for test accuracy runs* in case your prior session was killed partway through
                 workers=1,
                 tp=1,
                 quantized=False,
                 warmup=False):

        self.model_path = model_path or "mixtral/Mixtral-8x7B-Instruct-v0.1"
        self.workload_name = workload_name
        self.device = device

        if not batch_size:
            if device == "cpu":
                batch_size = 1
            else:
                batch_size = 32  # Reduce to 8 if using 4 GPUs, 16 for 8.
        self.batch_size = batch_size
        self.total_sample_count = total_sample_count
        self.tp = tp
        self.quantized=quantized
        self.warmup = warmup

        # dtype
        if dtype == 'bfloat16':
            self.amp_enabled = True
            self.amp_dtype = torch.bfloat16
        elif dtype == 'float16':
            self.amp_enabled = True
            self.amp_dtype = torch.float16
        else:
            self.amp_enabled = False
            self.amp_dtype = torch.float32

        self.dataset_path = dataset_path
        self.qsl = lg.ConstructQSL(self.total_sample_count, self.total_sample_count,
                                   self.LoadSamplesToRam, self.UnloadSamplesFromRam)

        self.num_workers = workers
        self.worker_threads = [None] * self.num_workers
        self.query_queue_list = [mp.JoinableQueue() for _ in range(self.num_workers)]
        self.query_queue_int = mp.Queue()
        self.output_queue = mp.Queue()
        self.alive_counter = mp.Value("i", 0)
        self.dead_counter = mp.Value("i", 0)
        self.cond_var = mp.Condition(lock=mp.Lock())

        self.use_cached_outputs = use_cached_outputs
        self.sample_counter = mp.Value("i", 0)
        self.current_counter_list = [mp.Value("i", 0) for _ in range(self.num_workers)]

        self.progress = None
        self.tp_sizes = []

    def LoadSamplesToRam(self, query_samples):
        pass

    def UnloadSamplesFromRam(self, query_samples):
        pass

    def load_dataset(self):
        if self.workload_name=="gptj":
            self.data_object = Dataset_CNN(self.dataset_path,
                                        model_checkpoint_path=self.model_path,
                                        total_sample_count=self.total_sample_count,
                                        pad_inputs=False)
            self.data_object.loadDataset()

        elif "mixtral" in self.workload_name:
            self.data_object = Dataset(self.model_path,
                                            dataset_path=self.dataset_path,
                                            total_sample_count=self.total_sample_count,
                                            device=self.device)
        else:
            self.data_object = Dataset(self.model_path,
                                        dataset_path=self.dataset_path,
                                        total_sample_count=self.total_sample_count,
                                        device=self.device)

    def start(self):
        # Create worker threads
        # To accommodate the awesome 43,42,43
        node_start_cores = utils.get_start_cores()
        core_lists = []
        if insts_per_node>0:
            for i in range(num_numa_nodes):
                for j in range(insts_per_node):
                    core_lists.append(list(range(node_start_cores[i]+j*cores_per_inst, node_start_cores[i]+(j+1)*cores_per_inst)))                    

        log.info("Loading dataset on main thread")
        self.load_dataset()

        for j in range(self.num_workers):
            core_list = []
            # for i in range(self.tp):
            core_list += core_lists[j*self.tp+0]

            tp_size_c = min(num_numa_nodes*insts_per_node - j*self.tp, self.tp)
            self.tp_sizes.append(tp_size_c)

            worker = Instance(
                model_path=self.model_path,
                workload_name=self.workload_name,
                dataset_path=self.dataset_path,
                device=self.device,
                batch_size=self.batch_size,
                total_sample_count=self.total_sample_count,
                rank=j,
                dtype=self.amp_dtype,
                core_list=tuple(core_list),
                node_list=tuple([math.floor(j*nodes_per_inst*self.tp)]),
                input_queue = self.query_queue_list[0],
                output_queue = self.output_queue,
                cond_var = self.cond_var,
                alive_counter = self.alive_counter,
                dead_counter = self.dead_counter,
                sample_counter = self.sample_counter,
                current_counter = self.current_counter_list[j],
                tp = tp_size_c,
                quantized=self.quantized,
                warmup=self.warmup
            )
            worker.start()
            self.worker_threads[j] = worker
            # Serialize worker tp initialization
            with self.cond_var:
                self.cond_var.wait_for(lambda: self.dead_counter.value == j+1)
                if self.tp>1:
                    # Ensure that tp nodes are correctly bound
                    time.sleep(20)

        with self.cond_var:
            print(f"Waiting for alive_counter to be equal to {self.num_workers}")
            self.cond_var.wait_for(lambda: self.alive_counter.value == self.num_workers)

        log.info(f"Starting internal issue query thread")
        query_thread = threading.Thread(target=self.issue_queries_int)
        query_thread.daemon = True
        query_thread.start()

        log.info(f"Starting Loadgen response thread")
        response_thread = threading.Thread(target=self.response_loadgen)
        response_thread.daemon = True
        response_thread.start()

    def stop(self):
        for i in range(self.num_workers):
            self.query_queue_list[i].put(None)

        if pbar:
            self.progress.close()

        for worker in self.worker_threads:
            worker.kill()

    def response_loadgen(self):
        keep_alive = True
        while keep_alive:
            result = self.output_queue.get()
            if result is None:
                keep_alive = False
            else:
                qid, processed_output = result

                for i in range(len(processed_output)):
                    n_tokens = processed_output[i].shape[0]
                    response_array = array.array("B", processed_output[i].tobytes())
                    bi = response_array.buffer_info()
                    response = [lg.QuerySampleResponse(qid[i], bi[0], bi[1], n_tokens)]
                    lg.QuerySamplesComplete(response)
                if pbar:
                    self.progress.update(len(processed_output))
                    self.progress.refresh()

    def get_sut(self):
        self.sut = lg.ConstructSUT(self.issue_queries, self.flush_queries)
        return self.sut

    def get_qsl(self):
        return self.qsl

    def predict(self,**kwargs):
        raise NotImplementedError

    def get_best_rank(self, value_added):
        current_counters = np.array([(self.current_counter_list[i].value+value_added)/max(self.tp_sizes[i]-0.8, 0.1) for i in range(self.num_workers)])
        target_rank = np.argmin(current_counters)
        return target_rank

    def issue_queries(self, query_samples):
        """ Receives samples from loadgen and adds them to queue. Users may choose to batch here"""
        if pbar and (self.progress is None):
            if len(query_samples)>1:
                self.progress = tqdm(total=len(query_samples), smoothing=0.0)
            else:
                self.progress = tqdm(smoothing=0.0)

        if len(query_samples)>1:
            # Offline
            log.info(f"IssueQuery started with {len(query_samples)} samples")
            for query in query_samples:
                self.query_queue_list[0].put(([query], [time.time()]))
        else:
            # Server
            # Static+continuous batching
            # print(f"IssueQuery issued 1 query")
            # self.query_queue_int.put((query_samples[0], time.time()))

            # Continuous batching
            for query in query_samples:
                with self.cond_var:
                    target_rank = self.get_best_rank(1)
                    self.current_counter_list[target_rank].value += 1
                self.query_queue_list[target_rank].put(([query], [time.time()]))

    def issue_queries_int(self):
        keep_alive = True
        query_lists = []
        time_compute_list = []
        time_start_lists = []
        time_left = 2
        time_limit_low = 0.4
        time_limit_high = 0.1
        while keep_alive:
            new_query = False
            try:
                query = self.query_queue_int.get(timeout=0.05)
            except:
                pass
            else:
                if query is None:
                    keep_alive = False
                    # print("Got None query")
                else:
                    query, start_time = query
                    query_ids = query.index
                    input_len = self.data_object.input_lens[query_ids]
                    time_compute_c = calculate_compute_latency(input_len)
                    start_time_c = time.time()
                    new_query = True
            # print(f"Getting new query or deciding on existing queries, keep_alive {keep_alive}, new_query {new_query}, len_query_list {len(query_lists)}")
            if new_query:
                # First query or empty lists
                if len(query_lists)==0:
                    query_lists.append([query])
                    time_compute_list.append(time_compute_c)
                    time_start_lists.append([start_time_c])
                else:
                    # print("Finding place to insert")
                    for i, query_list in enumerate(query_lists):
                        time_compute = time_compute_list[i]
                        time_wait = np.max(time.time()-np.array(time_start_lists[i]))
                        time_needed = time_compute + time_wait + time_compute_c
                        if time_needed<time_left-time_limit_high:
                            query_list.append(query)
                            time_compute_list[i] += time_compute_c
                            time_start_lists[i].append(start_time_c)
                            # print("Found a place to insert. Breaking")
                            break
                        # Cannot insert into existing lists
                        if i==len(query_lists)-1:
                            # print("Cannot find a place to insert. Appending")
                            query_lists.append([query])
                            time_compute_list.append(time_compute_c)
                            time_start_lists.append([start_time_c])
                            break
                        # if time_compute_c>1.1:
                        #     # Send long request to instance immediately
                        #     with self.cond_var:
                        #         target_rank = self.get_best_rank()
                        #         self.current_counter_list[target_rank].value += 1
                        #     print(f"Sending to rank {target_rank}, long query")
                        #     self.query_queue_list[target_rank].put(([query], [start_time]))
                        # else:
                        #     new_query = True
                        #     query_list.append(query)
                        #     time_start_list.append(start_time)
                        #     time_compute += time_compute_c

            # print("Deciding on dispatch")
            # for i, query_list in enumerate(query_lists):
            for i in range(len(query_lists)-1, -1, -1):
                query_list=query_lists[i]
                time_wait = np.max(time.time()-np.array(time_start_lists[i]))
                time_needed = time_compute_list[i] + time_wait
                # print(f"Traversing query_lists, {i}")
                if time_needed>time_left-time_limit_low:
                    with self.cond_var:
                        target_rank = self.get_best_rank(len(query_list))
                        self.current_counter_list[target_rank].value += len(query_list)
                    print(f"Sending to rank {target_rank}, length {len(query_list)}, wait_time {time_wait:.2f}s")
                    self.query_queue_list[target_rank].put((query_list, time_start_lists[i]))
                    del query_lists[i]
                    del time_compute_list[i]
                    del time_start_lists[i]

            # if len(query_list)>0:
            #     time_wait = np.max(time.time()-np.array(time_start_list))
            #     time_needed = time_compute + time_wait

            #     if new_query:
            #         if time_needed>time_left-time_limit_high:
            #             with self.cond_var:
            #                 target_rank = self.get_best_rank()
            #                 self.current_counter_list[target_rank].value += 1
            #             print(f"Sending to rank {target_rank}, query push total, wait time {time_wait:.2f}")
            #             self.query_queue_list[target_rank].put(([query_list[-1]], [time_start_list[-1]]))
            #             del query_list[-1]
            #             time_needed -= time_compute_c
            #             del time_start_list[-1]

            #     if time_needed>time_left-time_limit_low:
            #         # Send batched queries to instance
            #         with self.cond_var:
            #             target_rank = self.get_best_rank()
            #             self.current_counter_list[target_rank].value += len(query_list)
            #         print(f"Sending to rank {target_rank}, query reaching total, wait time {time_wait:.2f}")
            #         self.query_queue_list[target_rank].put((query_list, time_start_list))
            #         query_list = []
            #         time_compute = 0
            #         time_start_list = []

    def flush_queries(self):
        pass

    def __del__(self):
        pass


class SUTServer(SUT):
    def __init__(self,
                model_path=None,
                workload_name="llama2-70b",
                dtype="bfloat16",
                batch_size=None,
                device="cpu",
                total_sample_count=24576,
                dataset_path=None,
                workers=1,
                tp=1,
                quantized=False,
                warmup=False):
        super().__init__(
            model_path=model_path,
            workload_name=workload_name,
            dtype=dtype,
            batch_size=batch_size,
            device=device,
            total_sample_count=total_sample_count,
            dataset_path=dataset_path,
            workers=workers,
            tp=tp,
            quantized=quantized,
            warmup=warmup)
        self.first_token_queue = mp.Queue()

    def start(self):
        os.environ["OMP_NUM_THREADS"] = f"{cores_per_inst}"
        node_start_cores = utils.get_start_cores()
        core_lists = []
        if insts_per_node>0:
            for i in range(num_numa_nodes):
                for j in range(insts_per_node):
                    core_lists.append(list(range(node_start_cores[i]+j*cores_per_inst, node_start_cores[i]+(j+1)*cores_per_inst)))
        
        log.info("Loading dataset on main thread")
        self.load_dataset()

        # Create worker threads
        for j in range(self.num_workers):
            core_list = []
            core_list += core_lists[j*self.tp+0]

            tp_size_c = min(num_numa_nodes*insts_per_node - j*self.tp, self.tp)
            self.tp_sizes.append(tp_size_c)

            worker = Instance(
                model_path=self.model_path,
                workload_name=self.workload_name,
                dataset_path=self.dataset_path,
                device=self.device,
                batch_size=self.batch_size,
                total_sample_count=self.total_sample_count,
                rank=j,
                dtype=self.amp_dtype,
                core_list=tuple(core_list),
                node_list=tuple([math.floor(j*nodes_per_inst*self.tp)]),
                input_queue = self.query_queue_list[j],
                output_queue = self.output_queue,
                first_queue = self.first_token_queue,
                cond_var = self.cond_var,
                alive_counter = self.alive_counter,
                dead_counter = self.dead_counter,
                sample_counter = self.sample_counter,
                current_counter = self.current_counter_list[j],
                server=True,
                tp = tp_size_c,
                quantized=self.quantized,
                warmup=self.warmup
            )

            worker.start()
            self.worker_threads[j] = worker
            with self.cond_var:
                self.cond_var.wait_for(lambda: self.dead_counter.value == j+1)
                if self.tp>1:
                    time.sleep(20)

        with self.cond_var:
            
            self.cond_var.wait_for(lambda: self.alive_counter.value == self.num_workers)

        log.info(f"Starting internal issue query thread")
        query_thread = threading.Thread(target=self.issue_queries_int)
        query_thread.daemon = True
        query_thread.start()

        log.info(f"Starting Loadgen response thread")
        response_thread = threading.Thread(target=self.response_loadgen)
        response_thread.daemon = True
        response_thread.start()

        # Create first token response thread
        log.info(f"Starting first-token response thread")
        self.ft_response_thread = threading.Thread(target=self.process_first_tokens)
        self.ft_response_thread.daemon = True
        self.ft_response_thread.start()


    def process_first_tokens(self):

        while True:
            first_token_item = self.first_token_queue.get()

            if first_token_item is None:
                log.info("Exiting First token response thread")
                break

            qid, processed_output = first_token_item
            for i in range(len(processed_output)):
                response_data = array.array("B", processed_output[i].tobytes())
                bi = response_data.buffer_info()
                response = [lg.QuerySampleResponse(qid[i], bi[0], bi[1])]
                lg.FirstTokenComplete(response)
