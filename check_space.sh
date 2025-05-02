#!/bin/bash
# save as check_space.sh
FREE=$(df -k . | awk 'NR==2 {print $4}')
NEEDED=$((1500*1024))  # 1.5GB in KB

if [ $FREE -lt $NEEDED ]; then
    echo "Not enough free space! Need 1.5GB, have $(($FREE/1024))MB"
    exit 1
fi

# #./check_space.sh && python mlora_train.py \
#   --base_model TinyLlama/TinyLlama-1.1B-Chat-v0.4 \
#   --config ppo_critic_actor.yaml \
#   --device cuda \
#   --precision fp16 \
#   --metric_file experiment_logs