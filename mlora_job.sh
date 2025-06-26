#!/bin/bash
#SBATCH --job-name="mlora_pp"
#SBATCH --output="./slurm_logs/mlora_pp.%j.%t.out"
#SBATCH --error="./slurm_logs/mlora_pp.%j.%t.err"
#SBATCH --partition=gpuA40x4
#SBATCH --mem=32G
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=closest
#SBATCH --account=beis-delta-gpu
#SBATCH -t 00:02:00

module load cuda
module load anaconda3_gpu
module load gcc

source deactivate
source activate mlora_env_py312

# Set up distributed environment variables so that each process can communicate:
export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
export MASTER_PORT=12345

# Launch distributed training using srun with bash -c so that each process gets its own SLURM_PROCID.
srun --exclusive --gpus-per-node=1 --gpu-bind=closest --ntasks=2 bash -c '
    echo "Running on $(hostname), SLURM_PROCID is: $SLURM_PROCID, MASTER_ADDR is: $MASTER_ADDR";
    cd CS-598-FAL-Project
    python mlora_pp_train.py \
       --base_model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
       --config $HOME/CS-598-FAL-Project/demo/lora/lora_case_3.yaml \
       --pipeline \
       --device "cuda" \
       --rank $SLURM_PROCID \
       --nodes 2 \
       --no-recompute \
       --precision fp16
    '