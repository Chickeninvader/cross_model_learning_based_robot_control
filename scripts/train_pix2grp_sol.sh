#!/bin/bash
#SBATCH -N 1                    # number of nodes
#SBATCH -c 32                   # number of cores
#SBATCH -t 2-00:00:00           # time in d-hh:mm:ss
#SBATCH -p general              # partition
#SBATCH -q public               # QOS
#SBATCH --gres=gpu:a100:4       # GPU allocation
#SBATCH --mem=256G              # memory
#SBATCH -o slurm.%j.out         # file to save job's STDOUT (%j = JobId)
#SBATCH -e slurm.%j.err         # file to save job's STDERR (%j = JobId)
#SBATCH --mail-type=ALL         # Send an e-mail when a job starts, stops, or fails
#SBATCH --mail-user=kpham34@asu.edu
#SBATCH --export=NONE           # Purge the job-submitting shell environment
#SBATCH --job-name=pix2grp_train

echo "=========================================="
echo "Pix2Grp (CVPR 2024) Training Job"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Started: $(date)"
echo "=========================================="

# Load modules
module load mamba/latest

# Activate environment
source activate pix2grp

# Verify environment
echo "Python: $(which python)"
python --version
echo "PyTorch version:"
python -c "import torch; print(torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA devices:', torch.cuda.device_count())"

# GPU Information
echo "=========================================="
echo "GPU Information:"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv
echo "=========================================="

# Navigate to working directory
cd /scratch/kpham34/Pix2Grp_CVPR2024 || exit

# Set environment variables
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONPATH="${PWD}:${PYTHONPATH}"
export OMP_NUM_THREADS=8

# Training configuration
CONFIG=configs/psg/pix2grp_r50.yaml  # Adjust based on actual config
GPUS=4
PORT=$(shuf -i 10000-65535 -n 1)
DATASET=PSG  # PSG or VG dataset

echo "Training Configuration:"
echo "  Config: $CONFIG"
echo "  GPUs: $GPUS"
echo "  Port: $PORT"
echo "  Dataset: $DATASET"
echo "=========================================="

# Create log directory
mkdir -p logs

# Run distributed training
# Note: Adjust command based on actual Pix2Grp training script
python -m torch.distributed.launch \
  --nproc_per_node=$GPUS \
  --master_port=$PORT \
  tools/train_net.py \
  --config-file $CONFIG \
  --num-gpus $GPUS \
  OUTPUT_DIR ./outputs \
  SOLVER.IMS_PER_BATCH 16 \
  SOLVER.BASE_LR 0.01

echo "=========================================="
echo "Training completed: $(date)"
nvidia-smi --query-gpu=index,memory.used --format=csv
echo "=========================================="
