#!/usr/bin/env bash
cd ..

python main.py \
--model_name DELU_ACT \
--seed 0 \
--alpha_edl 1.2 \
--alpha_uct_guide 0.2 \
--amplitude 0.2 \
--alpha2 0.8 \
--rat_atn 5 \
--k 5 \
--interval 20 \
--dataset_name ActivityNet1.2 \
--path_dataset /path/to/CO2-ActivityNet-12 \
--num_class 100 \
--use_model DELU_ACT \
--dataset AntSampleDataset \
--lr 3e-5 \
--max_seqlen 60 \
--max_iter 22000
