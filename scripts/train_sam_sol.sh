#!/bin/bash
#SBATCH -N 1                    # number of nodes
#SBATCH -c 16                   # number of cores
#SBATCH -t 2-00:00:00           # time in d-hh:mm:ss
#SBATCH -p general              # partition
#SBATCH -q public               # QOS
#SBATCH --gres=gpu:a100:2       # GPU allocation
#SBATCH --mem=128G              # memory
#SBATCH -o slurm.%j.out         # file to save job's STDOUT (%j = JobId)
#SBATCH -e slurm.%j.err         # file to save job's STDERR (%j = JobId)
#SBATCH --mail-type=ALL         # Send an e-mail when a job starts, stops, or fails
#SBATCH --mail-user=kpham34@asu.edu
#SBATCH --export=NONE           # Purge the job-submitting shell environment
#SBATCH --job-name=sam_train

echo "=========================================="
echo "SAM Training Job"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Started: $(date)"
echo "=========================================="

# Load modules
module load mamba/latest

# Activate environment
source activate sam

# Verify environment
echo "Python: $(which python)"
python --version
echo "PyTorch version:"
python -c "import torch; print(torch.__version__); print('CUDA available:', torch.cuda.is_available())"

# GPU Information
echo "=========================================="
echo "GPU Information:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv
echo "=========================================="

# Navigate to working directory
cd /scratch/kpham34/segment-anything || exit

# Set environment variables
export CUDA_VISIBLE_DEVICES=0,1
export PYTHONPATH="${PWD}:${PYTHONPATH}"

# Training configuration
MODEL_TYPE=vit_b  # vit_b, vit_l, or vit_h
DATASET_PATH=/scratch/kpham34/data/sa1b  # SA-1B dataset
CHECKPOINT_DIR=./checkpoints
LOG_DIR=./logs

echo "Training Configuration:"
echo "  Model Type: $MODEL_TYPE"
echo "  Dataset: $DATASET_PATH"
echo "  Checkpoints: $CHECKPOINT_DIR"
echo "=========================================="

# Create directories
mkdir -p $CHECKPOINT_DIR
mkdir -p $LOG_DIR

# Run training (adjust based on SAM's training script)
python scripts/train.py \
  --model-type $MODEL_TYPE \
  --data-path $DATASET_PATH \
  --output-dir $CHECKPOINT_DIR \
  --batch-size 16 \
  --num-workers 8 \
  --lr 1e-4 \
  --epochs 50

echo "=========================================="
echo "Training completed: $(date)"
nvidia-smi --query-gpu=index,memory.used --format=csv
echo "=========================================="
