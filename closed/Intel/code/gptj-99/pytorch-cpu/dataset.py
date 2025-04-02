import os
from transformers import AutoTokenizer
import pandas as pd
import torch
import numpy as np
import pickle

import torch.nn.functional as F

import logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("MoE-Dataset")

class Dataset():
    def __init__(self, model_name=None, total_sample_count=15000, perf_count_override=None, dataset_path=None, device="cpu", pad_inputs=False):
        self.model_name = model_name or "mixtral/Mixtral-8x7B-Instruct-v0.1"
        self.dataset_path = dataset_path
        self.max_length = 2048
        self.device = device
        self.pad_inputs = pad_inputs

        self.load_tokenizer()
        self.load_processed_dataset()
        self.total_sample_count = min(len(self.input_ids), total_sample_count)
        self.perf_count = perf_count_override or self.total_sample_count

    def load_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            model_max_length=2048,
            padding_size="left",
            use_fast=False,)
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def load_processed_dataset(self):
        if not os.path.isfile(self.dataset_path):
            log.warn("Processed pickle file {} not found. Please check that the path is correct".format(self.dataset_path))
        
        processed_data = pd.read_pickle(self.dataset_path)
        input_tokens = processed_data['tok_input']
        
        self.input_ids = []
        self.input_lens = []
        self.attention_masks = []
        self.dataset = []
        for i,ids in enumerate(input_tokens):
            input_ids = torch.tensor(ids, dtype=torch.int32).view(1,-1).to(self.device)
            input_len = input_ids.shape[-1]
            attn_mask = torch.ones(input_len).view(1, input_len)
            if self.pad_inputs:
                pad_size = self.max_length - input_len
                pad_tup = (pad_size, 0)
                input_ids = F.pad(input_ids, pad=pad_tup, value=self.tokenizer.pad_token_id)
                attn_mask = F.pad(attn_mask, pad=pad_tup)

            self.input_ids.append(input_ids)
            self.input_lens.append(input_ids.size(1))
            self.dataset.append({'input_ids': input_ids, 'attention_mask': attn_mask})
            #self.attention_masks.append(None)

        print("Finished loading dataset")

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)

    def postProcess(self, out_tokens, input_seq_lens=None, query_id_list=None, sample_index_list=None, save=False):
        """ Postprocesses output prediction """
        #TODO: Create response object in postProcess(?)
        """
        preds = []
        for i in range(out_tokens.shape[0]):
            #pred = out_tokens[i].reshape(-1).cpu().numpy() # Slice up to original input length as below?

            input_len = input_seq_lens[i] if input_seq_lens else 0
            pred = out_tokens[i, input_len:].reshape(-1).cpu().numpy()
            preds.append(pred)
        """
        output_seqs = []
        
        for i,out_token in enumerate(out_tokens):
            output_seq = np.array(out_token).reshape(-1)
            while (len(output_seq)<=1):
                output_seq = np.append(output_seq, self.tokenizer.eos_token_id)
            output_seqs.append(output_seq)

            if save:
                query_id = [query_id_list[i]]
                # Save outputs
                if not os.path.exists("run_outputs"):
                    os.makedirs("run_outputs")
                fname = "q" + "_".join([str(i) for i in query_id])
                fname = f"run_outputs/{fname}.pkl"
                with open(fname, mode='wb') as f:
                    d = {"query_ids": query_id,
                        "outputs": output_seq}
                    print(f"Saving outputs to {fname}")
                    pickle.dump(d, f)

        return output_seqs

    def LoadSamplesToRam(self, sample_list):
        pass

    def UnloadSamplesToRam(self, sample_list):
        pass

    def __del__(self):
        pass
