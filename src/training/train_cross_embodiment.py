"""
Cross-Embodiment Policy Training Script

Main training script for Sol supercomputer (SLURM compatible).
Trains policy on mixed Franka + Sawyer demonstrations using scene graphs.
"""

import torch
import argparse
import yaml
from pathlib import Path
import logging
import sys
import os

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from policy.transformer_policy import TransformerPolicy
from policy.behavior_cloning import BCTrainer
from utils.data_loader import create_mixed_robot_dataloader, RobotDemonstrationDataset
from vision.scene_graph_embedder import SceneGraphEmbedder

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='Train cross-embodiment policy')

    # Data arguments
    parser.add_argument('--franka-data', type=str, nargs='+', required=True,
                       help='Paths to Franka demonstration data')
    parser.add_argument('--sawyer-data', type=str, nargs='+', required=True,
                       help='Paths to Sawyer demonstration data')
    parser.add_argument('--val-split', type=float, default=0.1,
                       help='Validation split fraction')

    # Model arguments
    parser.add_argument('--scene-graph-dim', type=int, default=256,
                       help='Scene graph embedding dimension')
    parser.add_argument('--hidden-dim', type=int, default=512,
                       help='Policy hidden dimension')
    parser.add_argument('--num-layers', type=int, default=4,
                       help='Number of transformer layers')
    parser.add_argument('--num-heads', type=int, default=8,
                       help='Number of attention heads')

    # Training arguments
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--num-epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--curriculum', action='store_true',
                       help='Use curriculum learning')
    parser.add_argument('--augment', action='store_true',
                       help='Use data augmentation')

    # Logging and checkpointing
    parser.add_argument('--log-dir', type=str, default='./logs',
                       help='Directory for logs and checkpoints')
    parser.add_argument('--wandb', action='store_true',
                       help='Use Weights & Biases logging')
    parser.add_argument('--save-freq', type=int, default=10,
                       help='Checkpoint save frequency (epochs)')

    # Hardware
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device for training')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of dataloader workers')

    # Resume training
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')

    # Config file
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config YAML file')

    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    args = parse_args()

    # Load config from file if provided
    if args.config:
        config = load_config(args.config)
        # Command line args override config file
        for key, value in vars(args).items():
            if value is not None:
                config[key] = value
    else:
        config = vars(args)

    logger.info("=" * 80)
    logger.info("Cross-Embodiment Policy Training")
    logger.info("=" * 80)
    logger.info(f"Configuration:\n{yaml.dump(config, default_flow_style=False)}")

    # Set device
    device = config['device'] if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")

    # Create log directory
    log_dir = Path(config['log_dir'])
    log_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(log_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f)

    # Initialize scene graph embedder
    logger.info("Initializing scene graph embedder...")
    scene_graph_embedder = SceneGraphEmbedder(
        embedding_dim=config['scene_graph_dim'],
        hidden_dim=config.get('gnn_hidden_dim', 256),
        num_layers=config.get('gnn_layers', 3),
    ).to(device)

    # Create data loaders
    logger.info("Loading demonstration data...")

    # Create train/val split
    import numpy as np
    np.random.seed(42)

    franka_paths = config['franka_data']
    sawyer_paths = config['sawyer_data']

    # Split data
    val_ratio = config['val_split']
    num_franka_val = max(1, int(len(franka_paths) * val_ratio))
    num_sawyer_val = max(1, int(len(sawyer_paths) * val_ratio))

    franka_indices = np.random.permutation(len(franka_paths))
    sawyer_indices = np.random.permutation(len(sawyer_paths))

    franka_train = [franka_paths[i] for i in franka_indices[num_franka_val:]]
    franka_val = [franka_paths[i] for i in franka_indices[:num_franka_val]]
    sawyer_train = [sawyer_paths[i] for i in sawyer_indices[num_sawyer_val:]]
    sawyer_val = [sawyer_paths[i] for i in sawyer_indices[:num_sawyer_val]]

    # Create datasets
    train_dataset = RobotDemonstrationDataset(
        data_paths=franka_train + sawyer_train,
        robot_types=['Franka'] * len(franka_train) + ['Sawyer'] * len(sawyer_train),
        scene_graph_embedder=scene_graph_embedder,
        normalize=True,
    )

    val_dataset = RobotDemonstrationDataset(
        data_paths=franka_val + sawyer_val,
        robot_types=['Franka'] * len(franka_val) + ['Sawyer'] * len(sawyer_val),
        scene_graph_embedder=scene_graph_embedder,
        normalize=True,
    )

    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Validation samples: {len(val_dataset)}")
    logger.info(f"Train statistics: {train_dataset.get_demo_statistics()}")

    # Initialize policy
    logger.info("Initializing transformer policy...")
    policy = TransformerPolicy(
        scene_graph_dim=config['scene_graph_dim'],
        robot_state_dim_franka=14,  # 7 joint pos + 7 joint vel
        robot_state_dim_sawyer=14,
        action_dim_franka=8,  # 7 joints + gripper
        action_dim_sawyer=8,
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        num_heads=config['num_heads'],
        dropout=config.get('dropout', 0.1),
        use_cross_attention=config.get('use_cross_attention', True),
    ).to(device)

    # Count parameters
    num_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    logger.info(f"Policy parameters: {num_params:,}")

    # Initialize trainer
    logger.info("Initializing BC trainer...")
    trainer_config = {
        'batch_size': config['batch_size'],
        'learning_rate': config['learning_rate'],
        'num_epochs': config['num_epochs'],
        'gradient_clip_norm': config.get('gradient_clip_norm', 1.0),
        'lr_scheduler': config.get('lr_scheduler', 'cosine'),
        'curriculum_learning': config.get('curriculum', False),
        'save_freq': config['save_freq'],
        'eval_freq': config.get('eval_freq', 5),
        'early_stopping_patience': config.get('early_stopping_patience', 20),
    }

    trainer = BCTrainer(
        policy=policy,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=trainer_config,
        device=device,
        log_dir=str(log_dir),
        use_wandb=config.get('wandb', False),
    )

    # Resume from checkpoint if specified
    if config.get('resume'):
        logger.info(f"Resuming from checkpoint: {config['resume']}")
        trainer.load_checkpoint(config['resume'])

    # Train
    logger.info("Starting training...")
    try:
        metrics = trainer.train()

        # Save final model
        final_path = log_dir / 'final_model.pth'
        torch.save({
            'model_state_dict': policy.state_dict(),
            'config': config,
            'metrics': metrics,
        }, final_path)

        logger.info(f"Training completed! Final model saved to {final_path}")

        # Plot training curves
        trainer.plot_training_curves(save_path=str(log_dir / 'training_curves.png'))

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        trainer._save_checkpoint('interrupted.pth')

    except Exception as e:
        logger.error(f"Training failed with error: {e}", exc_info=True)
        raise

    logger.info("=" * 80)
    logger.info("Training script completed")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
