
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

#2. on local environment: eg. agx
# python examples/piper_real/main.py --host 10.0.30.110 --port 8000
