# --config demo/lora/lora_case_1.yaml \
export MASTER_ADDR=c240g5-110127.wisc.cloudlab.us
export MASTER_PORT=12358
nsys profile --trace=cuda,nvtx,osrt --stats=true --output=mlora_profile --force-overwrite true \
python mlora_pp_train.py \
    --base_model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --trace \
    --config hetero_config.yaml \
    --device "cuda:0" \
    --rank 0 \
    --nodes 2 \
    --recompute \
    --precision fp16
