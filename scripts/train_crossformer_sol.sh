#!/bin/bash
#SBATCH -N 1                    # number of nodes
#SBATCH -c 32                   # number of cores
#SBATCH -t 5-00:00:00           # time in d-hh:mm:ss
#SBATCH -p general              # partition
#SBATCH -q public               # QOS
#SBATCH --gres=gpu:a100:4       # GPU allocation
#SBATCH --mem=256G              # memory
#SBATCH -o slurm.%j.out         # file to save job's STDOUT (%j = JobId)
#SBATCH -e slurm.%j.err         # file to save job's STDERR (%j = JobId)
#SBATCH --mail-type=ALL         # Send an e-mail when a job starts, stops, or fails
#SBATCH --mail-user=kpham34@asu.edu
#SBATCH --export=NONE           # Purge the job-submitting shell environment
#SBATCH --job-name=crossformer_train

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Started: $(date)"
echo "=========================================="

# Load mamba module
module load mamba/latest

# Activate crossformer environment
source activate crossformer

# Verify environment
echo "Python: $(which python)"
python --version
echo "PyTorch version:"
python -c "import torch; print(torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA devices:', torch.cuda.device_count())"
echo "Environment: $CONDA_DEFAULT_ENV"

# Verify GPU setup
echo "=========================================="
echo "GPU Information:"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv
echo "=========================================="

# Navigate to working directory
cd /scratch/kpham34/crossformer || exit

# Set environment variables
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONPATH="${PWD}:${PYTHONPATH}"
export OMP_NUM_THREADS=8

# Training configuration
DATA_PATH=/scratch/kpham34/data/bridge_data
SAVE_DIR=./checkpoints/crossformer

echo "Training Configuration:"
echo "  Data path: $DATA_PATH"
echo "  Save dir: $SAVE_DIR"
echo "  GPUs: 4"
echo "  Working dir: $PWD"
echo "=========================================="

# Run training with cross-embodiment data
python train.py \
  --data_path=$DATA_PATH \
  --save_dir=$SAVE_DIR \
  --batch_size=64 \
  --num_epochs=100 \
  --learning_rate=1e-4 \
  --num_workers=8 \
  --cross_embodiment=True \
  --num_gpus=4

echo "=========================================="
echo "Training completed: $(date)"
echo "Final GPU status:"
nvidia-smi --query-gpu=index,memory.used --format=csv
echo "=========================================="
