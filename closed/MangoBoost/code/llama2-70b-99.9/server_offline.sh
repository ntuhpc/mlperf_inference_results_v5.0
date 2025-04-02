#!/bin/bash
set -xeu

python3 server.py \
    -tp 1 \
    -dp 8 \
    -max_num_seqs 768 \
    --test_mode Offline
