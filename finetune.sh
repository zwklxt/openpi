#!/usr/bin/env bash

# This script is used to fine-tune a pre-trained model on a new task.for example, we can fine-tune the model on the piper.
# Note: you should not run this script directly, you should run the following command in the terminal to execute this script.
# Important:
# 1. this script not verify on huggingface, you should run it on the local machine.


#1.data preparation: translate the piper data format to the lerobot dataformat
# --raw_dir: the piper raw data  like *.hdf5 file list which to be converted
# --local_dir: the converted lerobot data will be saved in this directory
# --repo_id: the repo_id  for hug-face and assert directory
uv run examples/piper_real/convert_piper_data_to_lerobot.py \
    --raw_dir /home/anker/wenkai.zhang/repo/pi0/data/piper_raw/grab_up_move_down \
    --local_dir /home/anker/wenkai.zhang/repo/pi0/data/piper_lerobot/grab_up_move_down \
    --repo_id amigos-robot/grab_up_move_down

#piper-180
uv run examples/piper_real/convert_piper_data_to_lerobot.py \
    --raw_dir /home/anker/wenkai.zhang/repo/pi0/data/piper_180_raw \
    --local_dir /data01/wenkai.zhang/pi0/data/piper_lerobot/piper_180 \
    --repo_id amigos-robot/pick_up_bottle-180

#2. compute norm state for  the lerobot data which to be fine-tuned
# --config-name: the config name for the norm state,which comes from the training config file : openpi/training/config.pu
uv run scripts/compute_norm_stats.py --config-name pi0-piper-pick-bottle

#piper-180
uv run scripts/compute_norm_stats.py --config-name pi0-piper-pick-bottle-180

#3. train
# The following parameters are important in multi-GPU Device for fine-tuning:
#cuda_visible_devices set the GPU id to use for training
#export CUDA_VISIBLE_DEVICES=0,1,2,3

#xla_flags set the GPU id to use for training, if you use xla, you should set the xla_flags
#XLA_FLAGS='--xla_visible_devices=0,1,2,3'

#xla_python_client_mem_fraction set the GPU memory fraction to use for training, if you use xla, you should better set the xla_python_client_mem_fraction
#XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

#example for H100:pi0-base
CUDA_VISIBLE_DEVICES=2,4 \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run scripts/train.py pi0-piper-pick-bottle \
    --exp-name=pi0-base-pick-bottle \
    --save_interval=1000 \
    --batch_size=32 \
    --num_workers=2 \
    --overwrite \
    --wandb_enabled \

#example for H100:pi0-fast
CUDA_VISIBLE_DEVICES=5 \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run scripts/train.py pi0-fast-piper-pick-bottle \
    --exp-name=pi0-fast-pick-bottle \
    --save_interval=1000 \
    --batch_size=16 \
    --num_workers=2 \
    --overwrite \
    --wandb_enabled \


pi0-piper-pick-bottle-180
#example for H100:pi0-base
CUDA_VISIBLE_DEVICES=2,4 \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run scripts/train.py pi0-piper-pick-bottle-180 \
    --exp-name=pi0-piper-pick-bottle-180 \
    --save_interval=2000 \
    --batch_size=32 \
    --num_workers=2 \
    --overwrite \
    --wandb_enabled