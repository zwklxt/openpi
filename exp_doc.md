## Finetune experiment

### Task1: pick-up-bottle:random-grabbing-fixed-placement
#### 25.03.06
```bash
#1. conver data
uv run examples/piper_real/convert_piper_data_to_lerobot.py \
    --raw_dir /root/wenkai.zhang/data/piper_raw/pick-up-bottle-random-pose-100/ \
    --local_dir /HostData/wenkai.zhang/data/lerobot-format/pick-up-bottle-random-fixed-100/ \
    --repo_id amigos-robot/pi0-piper-pick-bottle-random-fixed-placement


#2. compute normal
CUDA_VISIBLE_DEVICES=4 uv run scripts/compute_norm_stats.py --config-name pi0-piper-pick-bottle-random-fixed-placement


#3. train
CUDA_VISIBLE_DEVICES=0,1\
XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run scripts/train.py pi0-piper-pick-bottle-random-fixed-placement \
    --name=pi0-piper-pick-bottle-random-fixed-placement \
    --project_name=openpi-pi0 \
    --exp-name=pi0-piper-pick-bottle-random-fixed-placement-compress \
    --checkpoint_base_dir=/HostData/wenkai.zhang/exp/ \
    --fsdp_devices=4 \
    --batch_size=48 \
    --num_workers=4 \
    --overwrite \
    --save_interval=2000 \
    --keep_period=4000 \
    --num_train_steps=30000

```


