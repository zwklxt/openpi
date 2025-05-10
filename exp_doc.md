## Finetune experiment

### Task1: pick-up-bottle:random-grabbing-fixed-placement
#### 25.03.06
```bash
#1. conver data
uv run examples/piper_real/convert_piper_data_to_lerobot.py \
    --raw_dir /root/wenkai.zhang/data/piper_raw/pick-up-bottle-random-pose-100/ \
    --local_dir /HostData/wenkai.zhang/data/lerobot-format/pick-up-bottle-random-fixed-100/ \
    --repo_id amigos-robot/pi0-piper-pick-bottle-random-fixed-placement \
    --task pick-bottle-random-fixed-placement 


#2. compute normal
CUDA_VISIBLE_DEVICES=4 uv run scripts/compute_norm_stats.py --config-name pi0-piper-pick-bottle-random-fixed-placement


#3. train
pi0_piper_pick-bottle-random-fixed-placement-lora

CUDA_VISIBLE_DEVICES=0,1 \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run scripts/train.py pi0-piper-pick-bottle-random-fixed-placement-lora \
    --name=pi0-piper-pick-bottle-random-fixed-placement \
    --project_name=openpi-pi0 \
    --exp-name=pi0-piper-pick-bottle-random-fixed-placement-lora-bs-32 \
    --checkpoint_base_dir=/HostData/wenkai.zhang/exp/ \
    --fsdp_devices=2 \
    --batch_size=32 \
    --num_workers=8 \
    --overwrite \
    --save_interval=5000 \
    --keep_period=4000 \
    --num_train_steps=60000

```
#### 25.03.09
```bash
#1. conver data

#2. compute normal

#3. train
CUDA_VISIBLE_DEVICES=4,5\
XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run scripts/train.py pi0-piper-pick-bottle-random-fixed-placement \
    --name=pi0-piper-pick-bottle-random-fixed-placement \
    --project_name=openpi-pi0 \
    --exp-name=pi0-piper-pick-bottle-random-fixed-placement-compress \
    --checkpoint_base_dir=/HostData/wenkai.zhang/exp/ \
    --fsdp_devices=2 \
    --batch_size=24 \
    --num_workers=4 \
    --overwrite \
    --save_interval=4000 \
    --keep_period=4000 \
    --num_train_steps=40000
```




### Task2: find and clean the ball
#### 25.03.07
```bash
#1. conver data
uv run examples/piper_real/convert_piper_data_to_lerobot.py \
    --raw_dir /HostData/amigos_shared/agilex_all_datasets/cleaning_the_ball/cleaning_the_ball_517mm/ \
    --local_dir /HostData/wenkai.zhang/data/lerobot-format/cleaning_the_ball_517mm_qpos/ \
    --repo_id amigos-robot/cleaning_the_ball_517mm_qpos


#2. compute normal
CUDA_VISIBLE_DEVICES=4 uv run scripts/compute_norm_stats.py --config-name pi0-piper-cleaning-the-ball-qpos


#3. train
CUDA_VISIBLE_DEVICES=4,5,6,7\
XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run scripts/train.py pi0-piper-cleaning-the-ball-qpos \
    --name=cleaning-the-ball\
    --project_name=openpi-pi0 \
    --exp-name=pi0-piper-cleaning-the-ball-517mm-qpos \
    --checkpoint_base_dir=/HostData/wenkai.zhang/exp/ \
    --fsdp_devices=4 \
    --batch_size=48 \
    --num_workers=4 \
    --overwrite \
    --save_interval=4000 \
    --keep_period=4000 \
    --num_train_steps=60000

```

```bash
#1. conver data
uv run examples/piper_real/convert_piper_data_to_lerobot.py \
    --raw_dir /HostData/amigos_shared/agilex_all_datasets/cleaning_the_ball/cleaning_the_ball_517mm/ \
    --local_dir /HostData/wenkai.zhang/data/lerobot-format/cleaning_the_ball_517mm/ \
    --repo_id amigos-robot/cleaning_the_ball_517mm


#2. compute normal
CUDA_VISIBLE_DEVICES=4 uv run scripts/compute_norm_stats.py --config-name pi0-piper-cleaning-the-ball


#3. train
XLA_PYTHON_CLIENT_PREALLOCATE=false \
CUDA_VISIBLE_DEVICES=0,1 \
uv run scripts/train.py pi0-piper-cleaning-the-ball \
    --name=cleaning-the-ball\
    --project_name=openpi-pi0 \
    --exp-name=pi0-piper-cleaning-the-ball-517mm \
    --checkpoint_base_dir=/HostData/wenkai.zhang/exp/ \
    --fsdp_devices=2 \
    --batch_size=24 \
    --num_workers=4 \
    --overwrite \
    --save_interval=4000 \
    --keep_period=4000 \
    --num_train_steps=40000

```
#### 25.03.09
```bash
#1. conver data
#2. compute normal
#3. train
XLA_PYTHON_CLIENT_PREALLOCATE=false \
CUDA_VISIBLE_DEVICES=6,7 \
uv run scripts/train.py pi0-piper-cleaning-the-ball-lora \
    --name=cleaning-the-ball\
    --project_name=openpi-pi0 \
    --exp-name=pi0-piper-cleaning-the-ball-517mm-lora \
    --checkpoint_base_dir=/HostData/wenkai.zhang/exp/ \
    --fsdp_devices=2 \
    --batch_size=24 \
    --num_workers=4 \
    --overwrite \
    --save_interval=4000 \
    --keep_period=4000 \
    --num_train_steps=40000
```


### Task3: agx-4000-multi-task
#### 25.03.11

```bash
#1. conver data
uv run examples/piper_real/convert_piper_data_to_lerobot_multitask.py \
    --raw_dir /HostData/amigos_shared/agilex_all_datasets/ \
    --local_dir /HostData/wenkai.zhang/data/lerobot-format/agx-4000-multi-task \
    --repo_id amigos-robot/agx-4000-multi-task


#2. compute normal
CUDA_VISIBLE_DEVICES=0 uv run scripts/compute_norm_stats.py --config-name pi0-piper-agx-4000-multi-task


#3. train
CUDA_VISIBLE_DEVICES=4,5,6,7 \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run scripts/train.py pi0-piper-agx-4000-multi-task \
    --name=pi0-piper-agx-4000\
    --project_name=openpi-pi0 \
    --exp-name=pi0-piper-agx-4000-multi-task-base \
    --checkpoint_base_dir=/HostData/wenkai.zhang/exp/ \
    --fsdp_devices=4 \
    --batch_size=128 \
    --num_workers=16 \
    --overwrite \
    --save_interval=4000 \
    --keep_period=4000 \
    --num_train_steps=60000
```

## 紧急任务 pi0实验 
### 4.21
#### pi0-base + 1000数据
```bash
CUDA_VISIBLE_DEVICES=0,1 \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run scripts/train.py pi0-piper-agx-1000-multi-task \
    --name=pi0-piper-agx-4000-stage1\
    --project_name=agx-4000 \
    --exp-name=pi0-piper-agx-1000-multi-task-base \
    --checkpoint_base_dir=/root/yujie.wei/pi0/exp/ \
    --fsdp_devices=2 \
    --batch_size=32 \
    --num_workers=16 \
    --save_interval=5000 \
    --num_train_steps=60000
```

#### pi0-base + 2000数据
```bash
CUDA_VISIBLE_DEVICES=2,3 \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run scripts/train.py pi0-piper-agx-2000-multi-task \
    --name=pi0-piper-agx-4000-stage1\
    --project_name=agx-4000 \
    --exp-name=pi0-piper-agx-2000-multi-task-base \
    --checkpoint_base_dir=/root/yujie.wei/pi0/exp/ \
    --fsdp_devices=2 \
    --batch_size=32 \
    --num_workers=16 \
    --save_interval=5000 \
    --num_train_steps=60000
```

#### pi0-base + 3000数据
```bash
CUDA_VISIBLE_DEVICES=4,5 \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run scripts/train.py pi0-piper-agx-3000-multi-task \
    --name=pi0-piper-agx-4000-stage1\
    --project_name=agx-4000 \
    --exp-name=pi0-piper-agx-3000-multi-task-base \
    --checkpoint_base_dir=/root/yujie.wei/pi0/exp/ \
    --fsdp_devices=2 \
    --batch_size=32 \
    --num_workers=16 \
    --save_interval=5000 \
    --num_train_steps=55000

尝试恢复训练——失败，因为保存的时候没有保存优化器和训练步数，没法恢复训练，必须重新训
```
### 4.22
#### pi0-base + 3000数据
```bash
CUDA_VISIBLE_DEVICES=4,5 \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run scripts/train.py pi0-piper-agx-3000-multi-task \
    --name=pi0-piper-agx-4000-stage1\
    --project_name=agx-4000 \
    --exp-name=pi0-piper-agx-3000-multi-task-base \
    --checkpoint_base_dir=/root/yujie.wei/pi0/exp/ \
    --fsdp_devices=2 \
    --batch_size=32 \
    --num_workers=16 \
    --save_interval=5000 \
    --overwrite \
    --num_train_steps=54000
```

#### pi0 base 1000 抓瓶子
```bash
CUDA_VISIBLE_DEVICES=0,1 \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run scripts/train.py pi0-piper-agx-1000-pick-and-place-bottle \
    --name=pi0-piper-agx-4000-stage2\
    --project_name=agx-4000 \
    --exp-name=pi0-piper-agx-1000-pick-and-place-bottle \
    --checkpoint_base_dir=/root/wenkai.zhang/exp/ \
    --fsdp_devices=2 \
    --batch_size=32 \
    --num_workers=16 \
    --save_interval=5000 \
    --num_train_steps=41000
```

#### pi0 base 2000 抓瓶子
```bash
CUDA_VISIBLE_DEVICES=2,3 \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run scripts/train.py pi0-piper-agx-2000-pick-and-place-bottle \
    --name=pi0-piper-agx-4000-stage2\
    --project_name=agx-4000 \
    --exp-name=pi0-piper-agx-2000-pick-and-place-bottle \
    --checkpoint_base_dir=/root/wenkai.zhang/exp/ \
    --fsdp_devices=2 \
    --batch_size=32 \
    --num_workers=16 \
    --save_interval=5000 \
    --num_train_steps=41000
```

#### pi0 base 1000 抓砖块 圆形
```bash
CUDA_VISIBLE_DEVICES=0,1 \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run scripts/train.py pi0-piper-agx-1000-pick-and-place-bricks \
    --name=pi0-piper-agx-4000-stage2\
    --project_name=agx-4000 \
    --exp-name=pi0-piper-agx-1000-pick-and-place-bricks \
    --checkpoint_base_dir=/root/wenkai.zhang/exp/ \
    --fsdp_devices=2 \
    --batch_size=32 \
    --num_workers=16 \
    --save_interval=5000 \
    --num_train_steps=41000
```

#### pi0 base 2000 抓砖块 圆形
```bash
CUDA_VISIBLE_DEVICES=2,3 \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run scripts/train.py pi0-piper-agx-2000-pick-and-place-bricks \
    --name=pi0-piper-agx-4000-stage2\
    --project_name=agx-4000 \
    --exp-name=pi0-piper-agx-2000-pick-and-place-bricks \
    --checkpoint_base_dir=/root/wenkai.zhang/exp/ \
    --fsdp_devices=2 \
    --batch_size=32 \
    --num_workers=16 \
    --save_interval=5000 \
    --num_train_steps=41000
```

### 4.23

#### pi0 base 3000 抓瓶子
```bash
CUDA_VISIBLE_DEVICES=2,3 \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run scripts/train.py pi0-piper-agx-3000-pick-and-place-bottle \
    --name=pi0-piper-agx-4000-stage2\
    --project_name=agx-4000 \
    --exp-name=pi0-piper-agx-3000-pick-and-place-bottle \
    --checkpoint_base_dir=/root/wenkai.zhang/exp/ \
    --fsdp_devices=2 \
    --batch_size=32 \
    --num_workers=16 \
    --save_interval=5000 \
    --num_train_steps=41000
```

#### pi0 base 3000 抓砖块 圆形
```bash
CUDA_VISIBLE_DEVICES=0,1 \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run scripts/train.py pi0-piper-agx-3000-pick-and-place-bricks \
    --name=pi0-piper-agx-4000-stage2\
    --project_name=agx-4000 \
    --exp-name=pi0-piper-agx-3000-pick-and-place-bricks \
    --checkpoint_base_dir=/root/wenkai.zhang/exp/ \
    --fsdp_devices=2 \
    --batch_size=32 \
    --num_workers=16 \
    --save_interval=5000 \
    --num_train_steps=41000
```

#### pi0 base 4000 抓瓶子
```bash
CUDA_VISIBLE_DEVICES=5,6 \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run scripts/train.py pi0-piper-agx-4000-pick-and-place-bottle \
    --name=pi0-piper-agx-4000-stage2\
    --project_name=agx-4000 \
    --exp-name=pi0-piper-agx-4000-pick-and-place-bottle \
    --checkpoint_base_dir=/root/wenkai.zhang/exp/ \
    --fsdp_devices=2 \
    --batch_size=32 \
    --num_workers=16 \
    --save_interval=5000 \
    --num_train_steps=41000
```

#### pi0 随机初始化
```bash
传输文件

/root/.cache/openpi/openpi-assets/checkpoints/pi0_base

scp -P 20040 -r "/root/.cache/openpi/openpi-assets/checkpoints/pi0_base" weiyujie@115.190.92.120:/file_system/vepfs/algorithm/yujie.wei/base_checkpoint

```

#### pi0-随机初始化 + 4000数据
```bash
CUDA_VISIBLE_DEVICES=0,1 \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run scripts/train.py pi0-piper-agx-random-4000-multi-task \
    --name=pi0-piper-agx-4000-stage1\
    --project_name=agx-4000 \
    --exp-name=pi0-piper-agx-random-4000-multi-task \
    --checkpoint_base_dir=/root/wenkai.zhang/exp/ \
    --fsdp_devices=2 \
    --batch_size=32 \
    --num_workers=16 \
    --save_interval=5000 \
    --num_train_steps=55000

```

### 4.24
#### pi0 随机初始化 4000 200 抓瓶子

```bash
CUDA_VISIBLE_DEVICES=4,5 \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run scripts/train.py pi0-piper-agx-random-init-4000-bottles \
    --name=pi0-piper-agx-4000-stage2 \
    --project_name=agx-4000 \
    --exp-name=pi0-piper-agx-random-init-4000-bottles \
    --checkpoint_base_dir=/root/wenkai.zhang/exp/ \
    --fsdp_devices=2 \
    --batch_size=32 \
    --num_workers=16 \
    --save_interval=5000 \
    --num_train_steps=55000 \
    --resume

CUDA_VISIBLE_DEVICES=4,5 \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run scripts/train.py pi0-piper-agx-4000-test \
    --name=pi0-piper-agx-4000-stage2 \
    --project_name=agx-4000 \
    --exp-name=pi0-piper-agx-random-init-4000-bottles \
    --checkpoint_base_dir=/root/wenkai.zhang/exp/ \
    --fsdp_devices=2 \
    --batch_size=32 \
    --num_workers=16 \
    --save_interval=5000 \
    --num_train_steps=55000 \
    --resume
```
#### pi0 随机初始化 4000 200 抓砖块

```bash
CUDA_VISIBLE_DEVICES=2,3 \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run scripts/train.py pi0-piper-agx-random-init-4000-bricks \
    --name=pi0-piper-agx-4000-stage2 \
    --project_name=agx-4000 \
    --exp-name=pi0-piper-agx-random-init-4000-bricks \
    --checkpoint_base_dir=/root/wenkai.zhang/exp/ \
    --fsdp_devices=2 \
    --batch_size=32 \
    --num_workers=16 \
    --save_interval=5000 \
    --num_train_steps=55000
```