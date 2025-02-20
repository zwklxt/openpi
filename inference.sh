
# This script is used to run the inference of the trained model
# Note: you should not run this script directly, you should run the following command in the terminal to execute this script.


#1. on service environment: eg. H100
# policy: checkpoint , mean infer policy  use checkpoint file
# --policy.config: the policy config name, which comes from the training config file : openpi/training/config.py
# --policy.dir: the checkpoint file path which to be used for inference
# Note: before runing ,you'd better to check to gpu device id ,and set the CUDA_VISIBLE_DEVICES
CUDA_VISIBLE_DEVICES=5 uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi0-piper-pick-bottle \
    --policy.dir=checkpoints/pi0-piper-pick-bottle/pi0-base-pick-bottle/19999

#piper-180
CUDA_VISIBLE_DEVICES=0 uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi0-piper-pick-bottle-180 \
    --policy.dir=/data01/wenkai.zhang/pi0/checkpoint/pi0-piper-pick-bottle-180/pi0-base-pick-bottle/10000 \


#2. on local environment: eg. agx
# --args.host STR        H100 (default: 10.0.30.110)     
# --args.port INT        (default: 8000)                 
# --args.action-horizon INT   (default: 50)   # the action horizon for the model                
# --args.num-episodes INT   (default: 1)                    
# --args.max-episode-steps INT   (default: 1000)                 
# --args.save-log, --args.no-save-log   (default: False)               

#set python path: if you run the script with the problem : cant find example/piper/..., you should set the python path
#export PYTHONPATH=$PYTHONPATH:/home/agilex/wenkai.zhang/Pi0/Pi0/openpi
python examples/piper_real/main.py \
    --args.host 10.0.30.110 \
    --args.port 8000 \
    --args.action-horizon 50 \
    --args.num-episodes 1 \
    --args.max-episode-steps 1000 \
    --args.save-log


