"""
Transformer-based Policy with Scene Graph Conditioning

Cross-embodiment policy architecture that uses scene graphs as morphology-agnostic
representations for transferring between Franka and Sawyer robots.

Architecture:
- Encoder: Scene graph embedding (256d from GNN)
- Decoder: Transformer layers with cross-attention
- Action heads: Robot-specific MLPs for Franka (8-DOF) and Sawyer (8-DOF)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List
import logging
import math

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer"""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
        """
        return x + self.pe[:x.size(1), :].unsqueeze(0)


class TransformerPolicy(nn.Module):
    """
    Transformer policy with scene graph conditioning

    Enables cross-embodiment transfer by using scene graphs as shared
    representation and robot-specific action heads for different morphologies.

    Args:
        scene_graph_dim: Dimension of scene graph embedding (from GNN)
        robot_state_dim: Dimension of robot proprioceptive state
        action_dim_franka: Action dimension for Franka (7 joints + 1 gripper)
        action_dim_sawyer: Action dimension for Sawyer (7 joints + 1 gripper)
        hidden_dim: Hidden dimension for transformer
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        dropout: Dropout probability
        use_cross_attention: Whether to use cross-attention for scene graph conditioning
    """

    def __init__(
        self,
        scene_graph_dim: int = 256,
        robot_state_dim_franka: int = 14,  # 7 joint pos + 7 joint vel
        robot_state_dim_sawyer: int = 14,  # 7 joint pos + 7 joint vel
        action_dim_franka: int = 8,  # 7 joints + gripper
        action_dim_sawyer: int = 8,  # 7 joints + gripper
        hidden_dim: int = 512,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_cross_attention: bool = True,
        activation: str = 'relu',
    ):
        super().__init__()

        self.scene_graph_dim = scene_graph_dim
        self.robot_state_dim_franka = robot_state_dim_franka
        self.robot_state_dim_sawyer = robot_state_dim_sawyer
        self.action_dim_franka = action_dim_franka
        self.action_dim_sawyer = action_dim_sawyer
        self.hidden_dim = hidden_dim
        self.use_cross_attention = use_cross_attention

        # Scene graph encoder
        self.scene_graph_encoder = nn.Sequential(
            nn.Linear(scene_graph_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Robot state encoders (robot-specific)
        self.robot_state_encoder_franka = nn.Sequential(
            nn.Linear(robot_state_dim_franka, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.robot_state_encoder_sawyer = nn.Sequential(
            nn.Linear(robot_state_dim_sawyer, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Positional encoding
        self.pos_encoder = PositionalEncoding(hidden_dim)

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation=activation,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Cross-attention layers (for scene graph conditioning)
        if use_cross_attention:
            self.cross_attention_layers = nn.ModuleList([
                nn.MultiheadAttention(
                    embed_dim=hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    batch_first=True,
                )
                for _ in range(num_layers)
            ])
            self.cross_attention_norms = nn.ModuleList([
                nn.LayerNorm(hidden_dim) for _ in range(num_layers)
            ])

        # Action heads (robot-specific)
        self.action_head_franka = self._build_action_head(hidden_dim, action_dim_franka, dropout)
        self.action_head_sawyer = self._build_action_head(hidden_dim, action_dim_sawyer, dropout)

        # Value heads (for RL if needed)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        self._init_weights()

        logger.info(
            f"TransformerPolicy initialized: "
            f"{num_layers} layers, {num_heads} heads, {hidden_dim}d hidden"
        )

    def _build_action_head(self, input_dim: int, action_dim: int, dropout: float) -> nn.Module:
        """Build robot-specific action head"""
        return nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.LayerNorm(input_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim // 2, input_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim // 4, action_dim),
            nn.Tanh(),  # Actions typically in [-1, 1]
        )

    def _init_weights(self):
        """Initialize weights with Xavier uniform"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(
        self,
        scene_graph_embedding: torch.Tensor,
        robot_state: torch.Tensor,
        robot_type: str,
        return_value: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass

        Args:
            scene_graph_embedding: Scene graph embedding (B, scene_graph_dim)
            robot_state: Robot proprioceptive state (B, state_dim)
            robot_type: 'Franka' or 'Sawyer'
            return_value: Whether to return value estimate

        Returns:
            actions: (B, action_dim) - Robot actions
            value: (B, 1) - Value estimate (if return_value=True)
        """
        batch_size = scene_graph_embedding.size(0)

        # Encode scene graph
        sg_encoded = self.scene_graph_encoder(scene_graph_embedding)  # (B, hidden_dim)

        # Encode robot state (robot-specific)
        if robot_type.lower() == 'franka' or robot_type.lower() == 'panda':
            robot_encoded = self.robot_state_encoder_franka(robot_state)
        elif robot_type.lower() == 'sawyer':
            robot_encoded = self.robot_state_encoder_sawyer(robot_state)
        else:
            raise ValueError(f"Unknown robot type: {robot_type}")

        # Combine scene graph and robot state as sequence
        # Shape: (B, 2, hidden_dim)
        sequence = torch.stack([sg_encoded, robot_encoded], dim=1)

        # Add positional encoding
        sequence = self.pos_encoder(sequence)

        # Transformer encoder with optional cross-attention
        if self.use_cross_attention:
            # Use cross-attention to condition on scene graph
            query = sequence
            key_value = sg_encoded.unsqueeze(1)  # (B, 1, hidden_dim)

            for i, (cross_attn, norm) in enumerate(zip(self.cross_attention_layers, self.cross_attention_norms)):
                # Self-attention via transformer
                query = self.transformer_encoder.layers[i](query)

                # Cross-attention to scene graph
                attn_output, _ = cross_attn(query, key_value, key_value)
                query = norm(query + attn_output)

            encoded = query
        else:
            # Standard transformer encoding
            encoded = self.transformer_encoder(sequence)

        # Pool sequence (use mean or last token)
        pooled = encoded.mean(dim=1)  # (B, hidden_dim)

        # Robot-specific action head
        if robot_type.lower() == 'franka' or robot_type.lower() == 'panda':
            actions = self.action_head_franka(pooled)
        elif robot_type.lower() == 'sawyer':
            actions = self.action_head_sawyer(pooled)
        else:
            raise ValueError(f"Unknown robot type: {robot_type}")

        if return_value:
            value = self.value_head(pooled)
            return actions, value
        else:
            return actions

    def get_action(
        self,
        scene_graph_embedding: torch.Tensor,
        robot_state: torch.Tensor,
        robot_type: str,
        deterministic: bool = True,
    ) -> torch.Tensor:
        """
        Get action for inference (wrapper around forward)

        Args:
            scene_graph_embedding: Scene graph embedding
            robot_state: Robot state
            robot_type: Robot type
            deterministic: Whether to use deterministic policy

        Returns:
            Action tensor
        """
        with torch.no_grad():
            action = self.forward(scene_graph_embedding, robot_state, robot_type, return_value=False)

        return action

    def compute_loss(
        self,
        scene_graph_embeddings: torch.Tensor,
        robot_states: torch.Tensor,
        actions_target: torch.Tensor,
        robot_types: List[str],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute behavior cloning loss

        Args:
            scene_graph_embeddings: Batch of scene graph embeddings
            robot_states: Batch of robot states
            actions_target: Target actions from demonstrations
            robot_types: List of robot types for each sample

        Returns:
            loss: Behavior cloning loss
            metrics: Dictionary of metrics
        """
        batch_size = scene_graph_embeddings.size(0)

        # Forward pass for each robot type
        actions_pred_list = []
        for i in range(batch_size):
            sg_emb = scene_graph_embeddings[i:i+1]
            robot_state = robot_states[i:i+1]
            robot_type = robot_types[i]

            action_pred = self.forward(sg_emb, robot_state, robot_type, return_value=False)
            actions_pred_list.append(action_pred)

        actions_pred = torch.cat(actions_pred_list, dim=0)

        # MSE loss for behavior cloning
        loss = F.mse_loss(actions_pred, actions_target)

        # Additional metrics
        with torch.no_grad():
            mae = F.l1_loss(actions_pred, actions_target)
            max_error = (actions_pred - actions_target).abs().max()

        metrics = {
            'bc_loss': loss.item(),
            'mae': mae.item(),
            'max_error': max_error.item(),
        }

        return loss, metrics


class MultiTaskTransformerPolicy(nn.Module):
    """
    Multi-task transformer policy for learning multiple manipulation tasks

    Extension of TransformerPolicy with task conditioning.
    """

    def __init__(
        self,
        num_tasks: int,
        task_embedding_dim: int = 64,
        **kwargs
    ):
        super().__init__()

        self.num_tasks = num_tasks
        self.task_embedding_dim = task_embedding_dim

        # Task embedding
        self.task_embeddings = nn.Embedding(num_tasks, task_embedding_dim)

        # Base policy
        kwargs['scene_graph_dim'] = kwargs.get('scene_graph_dim', 256) + task_embedding_dim
        self.policy = TransformerPolicy(**kwargs)

    def forward(
        self,
        scene_graph_embedding: torch.Tensor,
        robot_state: torch.Tensor,
        robot_type: str,
        task_id: torch.Tensor,
        return_value: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass with task conditioning

        Args:
            task_id: Task ID tensor (B,)
        """
        # Get task embedding
        task_emb = self.task_embeddings(task_id)  # (B, task_embedding_dim)

        # Concatenate with scene graph embedding
        sg_task = torch.cat([scene_graph_embedding, task_emb], dim=-1)

        # Forward through base policy
        return self.policy.forward(sg_task, robot_state, robot_type, return_value)


def test_transformer_policy():
    """Test transformer policy"""
    batch_size = 4
    scene_graph_dim = 256
    robot_state_dim = 14
    action_dim = 8

    # Create policy
    policy = TransformerPolicy(
        scene_graph_dim=scene_graph_dim,
        robot_state_dim_franka=robot_state_dim,
        robot_state_dim_sawyer=robot_state_dim,
        action_dim_franka=action_dim,
        action_dim_sawyer=action_dim,
        hidden_dim=512,
        num_layers=4,
        num_heads=8,
    )

    policy.eval()

    # Test Franka
    scene_graph_emb = torch.randn(batch_size, scene_graph_dim)
    robot_state = torch.randn(batch_size, robot_state_dim)

    actions_franka = policy(scene_graph_emb, robot_state, robot_type='Franka')
    print(f"Franka actions shape: {actions_franka.shape}")
    print(f"Franka actions range: [{actions_franka.min():.3f}, {actions_franka.max():.3f}]")

    # Test Sawyer
    actions_sawyer = policy(scene_graph_emb, robot_state, robot_type='Sawyer')
    print(f"Sawyer actions shape: {actions_sawyer.shape}")
    print(f"Sawyer actions range: [{actions_sawyer.min():.3f}, {actions_sawyer.max():.3f}]")

    # Test loss computation
    actions_target = torch.randn(batch_size, action_dim)
    robot_types = ['Franka', 'Sawyer', 'Franka', 'Sawyer']

    policy.train()
    loss, metrics = policy.compute_loss(scene_graph_emb, robot_state, actions_target, robot_types)
    print(f"Loss: {loss.item():.4f}")
    print(f"Metrics: {metrics}")

    # Count parameters
    num_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print(f"Total parameters: {num_params:,}")


if __name__ == "__main__":
    test_transformer_policy()
