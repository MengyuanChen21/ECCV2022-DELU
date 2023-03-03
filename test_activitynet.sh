#!/usr/bin/env bash
cd ..

python test.py
--dataset_name ActivityNet1.2 \
--dataset AntSampleDataset \
--num_class 100 \
--path_dataset /path/to/CO2-ActivityNet-12 \
--use_model DELU_ACT \
--model_name ActivityNet \
--max_seqlen 60 \
--without_wandb
