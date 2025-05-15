Install dependencies from pyproject.toml
## To run distributed heterogenous training
export MASTER_ADDR= "address of host GPU"
export MASTER_PORT=12345

On GPU 1:
```
python mlora_pp_train.py \
       --base_model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
       --config $HOME/CS-598-FAL-Project/demo/lora/lora_case_3.yaml \
       --pipeline \
       --device "cuda" \
       --rank 0 \
       --nodes 2 \
       --no-recompute \
       --precision fp16
```
On GPU 2:
```
python mlora_pp_train.py \
       --base_model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
       --config $HOME/CS-598-FAL-Project/demo/lora/lora_case_3.yaml \
       --pipeline \
       --device "cuda" \
       --rank 1 \
       --nodes 2 \
       --no-recompute \
       --precision fp16
```
## To run fLoRA
Run same MASTER_ADDR and MASTER_PORT setup
On GPU 1:
``` python mlora_pp_train.py \
      --base_model TinyLlama/TinyLlama-1.1B-Chat-v0.4 \
      --config demo/flora/flora_case_1.yaml \
      --pipeline \
      --device "cuda" \
      --rank 0 \
      --nodes 2 \
      --no-recompute \
      --precision fp32
```
On GPU 1:
``` python mlora_pp_train.py \
      --base_model TinyLlama/TinyLlama-1.1B-Chat-v0.4 \
      --config demo/flora/flora_case_1.yaml \
      --pipeline \
      --device "cuda" \
      --rank 1 \
      --nodes 2 \
      --no-recompute \
      --precision fp32
```
## To run profiling
Run same MASTER_ADDR and MASTER_PORT setup
On GPU 1:
```
nsys profile --trace=cuda,nvtx,osrt --stats=true --output=mlora_profile --force-overwrite true \
    python mlora_pp_train.py \
       --base_model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
       --config $HOME/CS-598-FAL-Project/demo/lora/lora_case_3.yaml \
       --trace \
       --pipeline \
       --device "cuda" \
       --rank 0 \
       --nodes 2 \
       --recompute \
       --precision fp16
```
On GPU 2:
```
nsys profile --trace=cuda,nvtx,osrt --stats=true --output=mlora_profile --force-overwrite true \
    python mlora_pp_train.py \
       --base_model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
       --config $HOME/CS-598-FAL-Project/demo/lora/lora_case_3.yaml \
       --trace \
       --pipeline \
       --device "cuda" \
       --rank 1 \
       --nodes 2 \
       --recompute \
       --precision fp16
```
## To Perform Heterogenous PPO Training for RLHF
```bash
python mlora_train.py --base_model TinyLlama/TinyLlama-1.1B-Chat-v0.4 --config ppo_critic_actor.yaml --device cuda --precision fp16 --metric_file experiment_logs
```
