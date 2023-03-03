#!/usr/bin/env bash
cd ..

python test.py \
--model_name delu_act \
--use_model DELU_ACT \
--path_dataset /path/to/CO2-ActivityNet-12 \
--dataset_name ActivityNet1.2 \
--dataset AntSampleDataset \
--num_class 100 \
--max_seqlen 60 \
--without_wandb
