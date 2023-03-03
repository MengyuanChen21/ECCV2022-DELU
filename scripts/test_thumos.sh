#!/usr/bin/env bash
cd ..

python test.py \
--model_name delu_thumos \
--dataset_name Thumos14reduced \
--path_dataset /path/to/CO2-THUMOS-14 \
--without_wandb