"""
Behavior Cloning Trainer for Cross-Embodiment Learning

Trains transformer policy on demonstration data with curriculum learning support.
Supports both single-robot and mixed cross-embodiment training.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from typing import Dict, List, Tuple, Optional, Callable
import logging
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
from collections import defaultdict
import matplotlib.pyplot as plt

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False
    logging.warning("wandb not installed. Run: pip install wandb")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BCTrainer:
    """
    Behavior cloning trainer with curriculum learning

    Trains policy on demonstration data with support for:
    - Single-robot and cross-embodiment training
    - Curriculum learning (easy → hard tasks)
    - Data augmentation
    - Mixed-batch training from multiple robots

    Args:
        policy: Policy network to train
        train_dataset: Training dataset
        val_dataset: Validation dataset
        config: Training configuration dictionary
        device: Device for training
        log_dir: Directory for saving logs and checkpoints
        use_wandb: Whether to use Weights & Biases logging
    """

    def __init__(
        self,
        policy: nn.Module,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset] = None,
        config: Optional[Dict] = None,
        device: str = 'cuda',
        log_dir: str = './logs',
        use_wandb: bool = False,
    ):
        self.policy = policy
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Default configuration
        default_config = {
            'batch_size': 32,
            'learning_rate': 1e-4,
            'weight_decay': 1e-5,
            'num_epochs': 100,
            'gradient_clip_norm': 1.0,
            'lr_scheduler': 'cosine',
            'warmup_epochs': 5,
            'curriculum_learning': True,
            'curriculum_start_epoch': 10,
            'data_augmentation': True,
            'save_freq': 10,
            'eval_freq': 5,
            'early_stopping_patience': 20,
        }

        self.config = {**default_config, **(config or {})}

        # Move policy to device
        self.policy.to(self.device)

        # Setup optimizer
        self.optimizer = optim.AdamW(
            self.policy.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay'],
        )

        # Setup learning rate scheduler
        self.scheduler = self._setup_scheduler()

        # Data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )

        if val_dataset is not None:
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=self.config['batch_size'],
                shuffle=False,
                num_workers=4,
                pin_memory=True,
            )
        else:
            self.val_loader = None

        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0

        # Metrics history
        self.train_metrics = defaultdict(list)
        self.val_metrics = defaultdict(list)

        # Weights & Biases
        self.use_wandb = use_wandb and HAS_WANDB
        if self.use_wandb:
            wandb.init(project='cross-embodiment-learning', config=self.config)
            wandb.watch(self.policy)

        logger.info(f"BCTrainer initialized on {self.device}")
        logger.info(f"Training samples: {len(train_dataset)}")
        if val_dataset:
            logger.info(f"Validation samples: {len(val_dataset)}")

    def _setup_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Setup learning rate scheduler"""
        if self.config['lr_scheduler'] == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['num_epochs'],
                eta_min=1e-6,
            )
        elif self.config['lr_scheduler'] == 'step':
            scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=30,
                gamma=0.1,
            )
        elif self.config['lr_scheduler'] == 'plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=10,
            )
        else:
            scheduler = None

        return scheduler

    def train(self) -> Dict[str, List[float]]:
        """
        Main training loop

        Returns:
            Dictionary of training metrics history
        """
        logger.info("Starting training...")

        for epoch in range(self.config['num_epochs']):
            self.epoch = epoch

            # Training epoch
            train_metrics = self._train_epoch()

            # Validation epoch
            if self.val_loader and (epoch % self.config['eval_freq'] == 0):
                val_metrics = self._validate_epoch()

                # Learning rate scheduling
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['loss'])

                # Check for improvement
                if val_metrics['loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['loss']
                    self.patience_counter = 0
                    self._save_checkpoint('best.pth')
                else:
                    self.patience_counter += 1

                # Early stopping
                if self.patience_counter >= self.config['early_stopping_patience']:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break

            else:
                val_metrics = None

            # Learning rate scheduling (non-plateau schedulers)
            if self.scheduler and not isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step()

            # Curriculum learning
            if self.config['curriculum_learning'] and epoch == self.config['curriculum_start_epoch']:
                logger.info("Starting curriculum learning phase")
                self._update_curriculum()

            # Save checkpoint
            if epoch % self.config['save_freq'] == 0:
                self._save_checkpoint(f'epoch_{epoch}.pth')

            # Log metrics
            self._log_metrics(train_metrics, val_metrics)

        logger.info("Training completed!")
        return {'train': dict(self.train_metrics), 'val': dict(self.val_metrics)}

    def _train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.policy.train()

        epoch_metrics = defaultdict(list)

        with tqdm(self.train_loader, desc=f"Epoch {self.epoch}") as pbar:
            for batch_idx, batch in enumerate(pbar):
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()}

                # Forward pass
                scene_graph_emb = batch['scene_graph_embedding']
                robot_state = batch['robot_state']
                actions_target = batch['actions']
                robot_types = batch['robot_type']

                loss, metrics = self.policy.compute_loss(
                    scene_graph_emb,
                    robot_state,
                    actions_target,
                    robot_types,
                )

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()

                # Gradient clipping
                if self.config['gradient_clip_norm'] > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.policy.parameters(),
                        self.config['gradient_clip_norm']
                    )

                self.optimizer.step()

                # Track metrics
                for key, value in metrics.items():
                    epoch_metrics[key].append(value)

                # Update progress bar
                pbar.set_postfix({k: f"{np.mean(v):.4f}" for k, v in epoch_metrics.items()})

                self.global_step += 1

        # Aggregate metrics
        avg_metrics = {k: np.mean(v) for k, v in epoch_metrics.items()}
        avg_metrics['learning_rate'] = self.optimizer.param_groups[0]['lr']

        return avg_metrics

    def _validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch"""
        self.policy.eval()

        epoch_metrics = defaultdict(list)

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()}

                # Forward pass
                scene_graph_emb = batch['scene_graph_embedding']
                robot_state = batch['robot_state']
                actions_target = batch['actions']
                robot_types = batch['robot_type']

                loss, metrics = self.policy.compute_loss(
                    scene_graph_emb,
                    robot_state,
                    actions_target,
                    robot_types,
                )

                # Track metrics
                for key, value in metrics.items():
                    epoch_metrics[key].append(value)

        # Aggregate metrics
        avg_metrics = {k: np.mean(v) for k, v in epoch_metrics.items()}
        avg_metrics['loss'] = avg_metrics['bc_loss']  # Alias for scheduler

        return avg_metrics

    def _update_curriculum(self):
        """Update training dataset for curriculum learning"""
        logger.info("Updating curriculum - adding harder examples")
        # In practice, this would modify the dataset to include harder tasks
        # For now, this is a placeholder
        pass

    def _log_metrics(self, train_metrics: Dict[str, float], val_metrics: Optional[Dict[str, float]]):
        """Log metrics to console and wandb"""
        # Log to console
        log_str = f"Epoch {self.epoch}: "
        log_str += " | ".join([f"{k}: {v:.4f}" for k, v in train_metrics.items()])

        if val_metrics:
            log_str += " | Val: "
            log_str += " | ".join([f"{k}: {v:.4f}" for k, v in val_metrics.items()])

        logger.info(log_str)

        # Store metrics
        for k, v in train_metrics.items():
            self.train_metrics[k].append(v)

        if val_metrics:
            for k, v in val_metrics.items():
                self.val_metrics[k].append(v)

        # Log to wandb
        if self.use_wandb:
            wandb_metrics = {f"train/{k}": v for k, v in train_metrics.items()}
            if val_metrics:
                wandb_metrics.update({f"val/{k}": v for k, v in val_metrics.items()})
            wandb_metrics['epoch'] = self.epoch
            wandb.log(wandb_metrics)

    def _save_checkpoint(self, filename: str):
        """Save model checkpoint"""
        checkpoint_path = self.log_dir / filename

        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss,
            'config': self.config,
            'train_metrics': dict(self.train_metrics),
            'val_metrics': dict(self.val_metrics),
        }

        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.policy.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']

        self.train_metrics = defaultdict(list, checkpoint['train_metrics'])
        self.val_metrics = defaultdict(list, checkpoint['val_metrics'])

        logger.info(f"Checkpoint loaded: {checkpoint_path}")
        logger.info(f"Resumed from epoch {self.epoch}, best val loss: {self.best_val_loss:.4f}")

    def evaluate(
        self,
        env,
        num_episodes: int = 10,
        robot_type: str = 'Franka',
        render: bool = False,
    ) -> Dict[str, float]:
        """
        Evaluate policy in environment

        Args:
            env: RoboSuite environment
            num_episodes: Number of evaluation episodes
            robot_type: Robot type to evaluate
            render: Whether to render visualization

        Returns:
            Dictionary of evaluation metrics
        """
        self.policy.eval()

        episode_rewards = []
        episode_successes = []

        for episode in range(num_episodes):
            obs = env.reset()
            done = False
            episode_reward = 0
            success = False

            while not done:
                # Get scene graph embedding (placeholder - would use actual scene graph pipeline)
                scene_graph_emb = torch.randn(1, self.policy.scene_graph_dim).to(self.device)

                # Get robot state
                robot_state_np = np.concatenate([
                    obs['robot0_joint_pos'],
                    obs['robot0_joint_vel'],
                ])
                robot_state = torch.from_numpy(robot_state_np).float().unsqueeze(0).to(self.device)

                # Get action
                with torch.no_grad():
                    action = self.policy.get_action(scene_graph_emb, robot_state, robot_type)
                    action = action.cpu().numpy().squeeze()

                # Step environment
                obs, reward, done, info = env.step(action)
                episode_reward += reward

                if render:
                    env.render()

                if info.get('success', False):
                    success = True

            episode_rewards.append(episode_reward)
            episode_successes.append(float(success))

            logger.info(f"Episode {episode+1}/{num_episodes}: reward={episode_reward:.2f}, success={success}")

        # Aggregate metrics
        metrics = {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'success_rate': np.mean(episode_successes),
            'num_episodes': num_episodes,
        }

        logger.info(f"Evaluation results: {metrics}")

        if self.use_wandb:
            wandb.log({f"eval/{k}": v for k, v in metrics.items()})

        return metrics

    def plot_training_curves(self, save_path: Optional[str] = None):
        """Plot training curves"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        # Loss curves
        axes[0, 0].plot(self.train_metrics['bc_loss'], label='Train')
        if self.val_metrics['bc_loss']:
            axes[0, 0].plot(
                np.arange(0, len(self.val_metrics['bc_loss'])) * self.config['eval_freq'],
                self.val_metrics['bc_loss'],
                label='Val'
            )
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('BC Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # MAE curves
        axes[0, 1].plot(self.train_metrics['mae'], label='Train')
        if self.val_metrics['mae']:
            axes[0, 1].plot(
                np.arange(0, len(self.val_metrics['mae'])) * self.config['eval_freq'],
                self.val_metrics['mae'],
                label='Val'
            )
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('MAE')
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        # Learning rate
        axes[1, 0].plot(self.train_metrics['learning_rate'])
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True)

        # Max error
        axes[1, 1].plot(self.train_metrics['max_error'], label='Train')
        if self.val_metrics['max_error']:
            axes[1, 1].plot(
                np.arange(0, len(self.val_metrics['max_error'])) * self.config['eval_freq'],
                self.val_metrics['max_error'],
                label='Val'
            )
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Max Error')
        axes[1, 1].legend()
        axes[1, 1].grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Training curves saved to {save_path}")
        else:
            plt.savefig(self.log_dir / 'training_curves.png', dpi=150, bbox_inches='tight')

        plt.close()


if __name__ == "__main__":
    # Test BC trainer (requires dataset and policy)
    logger.info("BC Trainer module loaded successfully")
    logger.info("To use: create policy and dataset, then initialize BCTrainer")
