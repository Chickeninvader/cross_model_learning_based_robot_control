#!/bin/bash
#SBATCH -N 1                    # number of nodes
#SBATCH -c 64                   # number of cores
#SBATCH -t 7-00:00:00           # time in d-hh:mm:ss
#SBATCH -p general              # partition
#SBATCH -q public               # QOS
#SBATCH --gres=gpu:a100:8       # GPU allocation
#SBATCH --mem=512G              # memory
#SBATCH -o slurm.%j.out         # file to save job's STDOUT (%j = JobId)
#SBATCH -e slurm.%j.err         # file to save job's STDERR (%j = JobId)
#SBATCH --mail-type=ALL         # Send an e-mail when a job starts, stops, or fails
#SBATCH --mail-user=kpham34@asu.edu
#SBATCH --export=NONE           # Purge the job-submitting shell environment
#SBATCH --job-name=octo_train

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Started: $(date)"
echo "=========================================="

# Load mamba module
module load mamba/latest

# Activate octo environment
source activate octo

# Verify environment
echo "Python: $(which python)"
python --version
echo "JAX version:"
python -c "import jax; print(jax.__version__); print('Devices:', jax.devices())"
echo "Environment: $CONDA_DEFAULT_ENV"

# Verify GPU setup
echo "=========================================="
echo "GPU Information:"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv
echo "=========================================="

# Navigate to working directory
cd /scratch/kpham34/octo || exit

# Set environment variables
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONPATH="${PWD}:${PYTHONPATH}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.8

# Training configuration
DATA_PATH=/scratch/kpham34/data/oxe_data
CONFIG=configs/finetune_config.py
SAVE_DIR=./checkpoints/octo_finetuned

echo "Training Configuration:"
echo "  Data path: $DATA_PATH"
echo "  Config: $CONFIG"
echo "  Save dir: $SAVE_DIR"
echo "  GPUs: 8"
echo "  Working dir: $PWD"
echo "=========================================="

# Run training with diffusion policy
python scripts/finetune.py \
  --config=$CONFIG \
  --config.data_path=$DATA_PATH \
  --config.save_dir=$SAVE_DIR \
  --config.batch_size=256 \
  --config.num_steps=100000 \
  --config.learning_rate=3e-4

echo "=========================================="
echo "Training completed: $(date)"
echo "Final GPU status:"
nvidia-smi --query-gpu=index,memory.used --format=csv
echo "=========================================="
