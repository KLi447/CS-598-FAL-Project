# MLoRA Reproduce Instructions

## Experimental Setup

* download the base model to the experimental platforms: [TinyLlama-1.1B](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0) / [Llama2-7B](https://huggingface.co/meta-llama/Llama-2-7b-hf) / [Llama2-13B](https://huggingface.co/meta-llama/Llama-2-13b-hf) / [Llama2-70B](https://huggingface.co/meta-llama/Llama-2-70b-hf)
* Follow the [Quickstart guide](https://github.com/TUDB-Labs/mLoRA?tab=readme-ov-file#quickstart) in README.md to install mLoRA (**NOTE**: before installation, switch to the reproduction branch by running `git fetch && git checkout -t origin/paper_reproduce`)
* Follow the [Installation Guide](https://docs.nvidia.com/nsight-systems/InstallationGuide/index.html#) in nsight-systems documentation to install nsys tools for profiling BatchLoRA operator performance

## End-to-End Test

*NOTE: those scripts will output the performence metrics.*

### evaluate the performance of TP

single-machine, multi-GPU (4 * A6000 / 1)：
on single machine, launch the following command *4* times simultaneously with different `rank` and `local_rank` values.

* Set `master_node_ip` to `localhost`.
* Set `world_size` to `4`.
* Set `base_model_path` to the path of the downloaded base model.
* Use `rank` values of `0, 1, 2, 3` respectively.
* Set the `local_rank` same as the `rank` value.

multi-machine, multi-GPU (8 * 3090 / 8):
on 8 machines, simultaneously launch the following command on each machine with different `rank` values.

* Set `master_node_ip` to the IP address of one of the machines.
* Set `world_size` to `8`. Set `base_model_path` to the path of the downloaded base model.
* Use `rank` values of `0, 1, 2, 3, 4, 5, 6, 7` respectively.
* Set the `local_rank` to 0.

```bash
MASTER_ADDR=<master_node_ip> MASTER_PORT=17771 \
LOCAL_RANK=<local_rank> RANK=<rank> WORLD_SIZE=<world_size> \
python benchmarks/bench_peft_tp.py \
    --base_model <base_model_path> \
    --batch_size 8
```

### evaluate the performance of FSDP

single-machine, multi-GPU (4 * A6000 / 1)：
on single machine, launch the following command *4* times simultaneously with different `rank` and `local_rank` values.

* Set `master_node_ip` to `localhost`.
* Set `world_size` to `4`. Set `base_model_path` to the path of the downloaded base model.
* Use `rank` values of `0, 1, 2, 3` respectively.
* Set the `local_rank` same as the `rank` value.

multi-machine, multi-GPU (8 * 3090 / 8):
on 8 machines, simultaneously launch the following command on each machine with different `rank` values.

* Set `master_node_ip` to the IP address of one of the machines.
* Set `world_size` to `8`. S
* et `base_model_path` to the path of the downloaded base model.
Use `rank` values of `0, 1, 2, 3, 4, 5, 6, 7` respectively, and set the `local_rank` to 0.

```bash
MASTER_ADDR=<master_node_ip> MASTER_PORT=17771 \
LOCAL_RANK=<local_rank> RANK=<rank> WORLD_SIZE=<world_size> \
python benchmarks/bench_peft_tp.py \
    --base_model <base_model_path> \
    --batch_size 8
```

### evaluate the performance of GPipe

single-machine, multi-GPU (4 * A6000 / 1)：
on single machine, launch the following command *4* times simultaneously with different `rank` and `device` values.

* Set `master_node_ip` to `localhost`.
* Set `base_model_path` to the path of the downloaded base model.
* Use `rank` values of `0, 1, 2, 3` respectively.
* Use `device` values of `cuda:0, cuda:1, cuda:2, cuda:3` respectively.
* For TinyLlama-1.1B model, set `balance` to `6 5 6 8`.
* For Llama2-7B model, set `balance` to `9 8 8 10`.
* For Llama2-13B model, set `balance` to `11 10 10 12`.
* For Llama2-70B model, set `balance` to `21 20 20 22`.

multi-machine, multi-GPU (8 * 3090 / 8):
on 8 machines, simultaneously launch the following command on each machine with different `rank` and `device` values.

* Set `master_node_ip` to the IP address of one of the machines.
* Set `base_model_path` to the path of the downloaded base model.
* Use `rank` values of `0, 1, 2, 3, 4, 5, 6, 7` respectively.
* Set `device` to `cuda:0`.
* For TinyLlama-1.1B model, set `balance` to `3 3 3 3 3 3 3 4`.
* For Llama2-7B model, set `balance` to `5 4 4 4 4 4 4 6`.
* For Llama2-13B model, set `balance` to `6 5 5 5 5 5 5 7`.
* For Llama2-70B model, set `balance` to `11 10 10 10 10 10 10 12`.

```bash
MASTER_ADDR=<master_node_ip> MASTER_PORT=17771 \
BATCH_SIZE=8 LORA_RANK=16 TASK=1 \
python benchmarks/bench_mlora_pp.py \
    --base_model <base_model_path> \
    --device <device> \
    --precision fp32 \
    --pipeline \
    --rank <rank> \
    --balance <balance_list> \
    --config demo/lora/lora_case_1.yaml \
    --no-recompute
```

### evaluate the performance of mLoRA

Consistent with testing GPipe, we start with `1` and gradually increase `task_cnt` to test the number of LoRAs that can be fine-tuned simultaneously and their throughput under different modes.

single-machine, multi-GPU (4 * A6000 / 1)：
on single machine, launch the following command *4* times simultaneously with different `rank` and `device` values.

* Set `master_node_ip` to `localhost`.
* Set `base_model_path` to the path of the downloaded base model.
* Use `rank` values of `0, 1, 2, 3` respectively.
* Use `device` values of `cuda:0, cuda:1, cuda:2, cuda:3` respectively.
* For TinyLlama-1.1B model, set `balance` to `6 5 6 8`.
* For Llama2-7B model, set `balance` to `9 8 8 10`.
* For Llama2-13B model, set `balance` to `11 10 10 12`.
* For Llama2-70B model, set `balance` to `21 20 20 22`.

multi-machine, multi-GPU (8 * 3090 / 8):
on 8 machines, simultaneously launch the following command on each machine with different `rank` and `device` values.

* Set `master_node_ip` to the IP address of one of the machines.
* Set `base_model_path` to the path of the downloaded base model.
* Use `rank` values of `0, 1, 2, 3, 4, 5, 6, 7` respectively.
* Set `device` to `cuda:0`.
* For TinyLlama-1.1B model, set `balance` to `3 3 3 3 3 3 3 4`.
* For Llama2-7B model, set `balance` to `5 4 4 4 4 4 4 6`.
* For Llama2-13B model, set `balance` to `6 5 5 5 5 5 5 7`.
* For Llama2-70B model, set `balance` to `11 10 10 10 10 10 10 12`.

```bash
MASTER_ADDR=<master_node_ip> MASTER_PORT=17771 \
BATCH_SIZE=8 LORA_RANK=16 TASK=<task_cnt> \
python benchmarks/bench_mlora_pp.py \
    --base_model <base_model_path> \
    --device <device> \
    --precision fp32 \
    --pipeline \
    --rank <rank> \
    --balance <balance_list> \
    --config demo/lora/lora_case_1.yaml \
    --no-recompute
```

single-machine, single-GPU (A6000 / 1)：
on single machine, run the script below:

* Set `base_model_path` to the path of the downloaded base model.
* We start with `1` and gradually increase `lora_num` to test the number of LoRAs that can be fine-tuned simultaneously and their throughput.

```bash
BATCH_SIZE=8 TASK=<lora_num> \
python benchmarks/bench_mlora_batchlora.py \
    --base_model <base_model_path> \
    --config demo/lora/lora_case_1.yaml \
    --precision int8
```

### evaluate the performance of PEFT

single-machine, single-GPU (A6000 / 1):
on single machine, run the script below:

* Set `base_model_path` to the path of the downloaded base model.
* We start with `1` and gradually increase `lora_num` to test the number of LoRAs that can be fine-tuned simultaneously and their throughput.

```bash
python benchmarks/bench_peft_batchlora.py \
   --base_model <base_model_path> \
   --load_8bit \
   --seq_len 512 \
   --batch_size 8 \
   --lora_cnt <lora_num> \
   --peft_mode switch 
```

## Effectiveness of LoRAPP

### Effectiveness of overlapping communication

the command for test the effectiveness of overlapping is the same as testing mLoRA performance, but you need to add the environment variable `BLOCK`.

* Set `BLOCK=true`, the overlapping feature is disabled.

```bash
BLOCK=true MASTER_ADDR=<master_node_ip> MASTER_PORT=17771 \
BATCH_SIZE=8 LORA_RANK=16 TASK=<task_cnt> \
python benchmarks/bench_mlora_pp.py \
    --base_model <base_model_path> \
    --device <device> \
    --precision fp32 \
    --pipeline \
    --rank <rank> \
    --balance <balance_list> \
    --config demo/lora/lora_case_1.yaml \
    --no-recompute
```

### Scalability

the command for test the scalability of mLoRA is the same as testing mLoRA performance, but you need to modify the environment variable `LORA_RANK`.

```bash
BLOCK=true MASTER_ADDR=<master_node_ip> MASTER_PORT=17771 \
BATCH_SIZE=8 LORA_RANK=<lora_rank> TASK=<task_cnt> \
python benchmarks/bench_mlora_pp.py \
    --base_model <base_model_path> \
    --device <device> \
    --precision fp32 \
    --pipeline \
    --rank <rank> \
    --balance <balance_list> \
    --config demo/lora/lora_case_1.yaml \
    --no-recompute
```

## Effectiveness of BatchLoRA

*NOTE*: need to patch `peft` and `transformer` to profile performance, we put the patch in the directory `scripts/patch`, and you should follow the instructions to patch them.

```bash
pip install peft==0.10.0
pip install transformers==4.38.2

patch -p0 <the_path_of_python>/lib/python3.12/site-packages/peft/tuners/lora/bnb.py ./scripts/patch/peft_v_0_10_0/tuners_lora_bnb.patch
patch -p0 <the_path_of_python>/lib/python3.12/site-packages/peft/tuners/lora/layer.py ./scripts/patch/peft_v_0_10_0/tuners_lora_layer.patch
patch -p0 <the_path_of_python>/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py ./scripts/patch/transformers_v_4_38_2/models_llama_modeling_llama.patch
```

single-machine, single-GPU (A6000 / 1)：
on single machine, run the script below to profile batchlora and peft:

* We use the nsys and the nvtx api to trace and record the time
* Set `base_model_path` to the path of the downloaded base model.
* We start with `1` and gradually increase `lora_num` to test the number of LoRAs that can be fine-tuned simultaneously and their throughput.

```bash
nsys profile -w true -t cuda,nvtx -s none -o <file_name> -f true -x true \
python benchmarks/bench_peft_batchlora.py \
   --base_model <base_model_path> \
   --load_8bit \
   --seq_len 512 \
   --batch_size 8 \
   --lora_cnt <lora_num> \
   --peft_mode switch 
```

```bash
BATCH_SIZE=8 TASK=<lora_num> \
nsys profile -w true -t cuda,nvtx -s none -o <file_name> -f true -x true \
python benchmarks/bench_mlora_batchlora.py \
    --base_model <base_model_path> \
    --config demo/lora/lora_case_1.yaml \
    --precision int8
```

and then use `nsys export` tool to generate the metric database file for analyze:
```bash
nsys export --type sqlite --output <db_file_name> <file_name>.nsys-rep
```

and then use our scripts to get the report:
```bash
python scripts/performance_report.py --db <db_file_name> --output <report_file_name>.csv
```

The first 5 columns in the report_file represent: event_name | call_api_start_time | call_api_end_time | kernel_start_time | kernel_end_time

You can calculate:
```txt
Kernel execution time = kernel_end_time - kernel_start_time
Kernel launch time = Training time - kernel_computation_time
```
