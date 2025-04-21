export MASTER_ADDR=c240g5-110205.wisc.cloudlab.us
export MASTER_PORT=12355
nsys profile --trace=cuda,nvtx --output=mlora_profile --force-overwrite true \
python mlora_train.py \
    --base_model TinyLlama/TinyLlama-1.1B-Chat-v0.4 \
    --trace \
    --config demo/lora/lora_case_1.yaml \
    --device "cuda:0" \
    --rank 0 \
    --balance 25 \
    --no-recompute \
    --precision fp32