"""
Scene Graph Embedding with Graph Neural Networks

Converts scene graphs to fixed-size embeddings for policy conditioning.
Uses GNN-based encoder to preserve spatial and relational information.

Embedding dimension: 256-512d for efficient policy learning.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
import logging

try:
    import torch_geometric
    from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_max_pool
    from torch_geometric.data import Data, Batch
    HAS_TORCH_GEOMETRIC = True
except ImportError:
    HAS_TORCH_GEOMETRIC = False
    logging.warning("torch_geometric not installed. Install with: pip install torch-geometric")

try:
    from scene_graph_generator import SceneGraph, SceneGraphNode, SceneGraphEdge
except ImportError:
    from .scene_graph_generator import SceneGraph, SceneGraphNode, SceneGraphEdge

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SceneGraphEmbedder(nn.Module):
    """
    GNN-based scene graph encoder

    Converts scene graphs to fixed-size embeddings using Graph Convolutional Networks.
    Preserves both node features (objects) and edge features (relationships).

    Args:
        node_feature_dim: Dimension of input node features
        edge_feature_dim: Dimension of edge features
        hidden_dim: Hidden layer dimension
        embedding_dim: Output embedding dimension (256 or 512)
        num_layers: Number of GNN layers
        gnn_type: Type of GNN ('gcn', 'gat', 'gin')
        pooling: Graph pooling method ('mean', 'max', 'attention')
        dropout: Dropout probability
    """

    def __init__(
        self,
        node_feature_dim: int = 128,
        edge_feature_dim: int = 32,
        hidden_dim: int = 256,
        embedding_dim: int = 256,
        num_layers: int = 3,
        gnn_type: str = 'gat',
        pooling: str = 'mean',
        dropout: float = 0.1,
    ):
        super().__init__()

        if not HAS_TORCH_GEOMETRIC:
            raise ImportError("torch_geometric required for SceneGraphEmbedder")

        self.node_feature_dim = node_feature_dim
        self.edge_feature_dim = edge_feature_dim
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.gnn_type = gnn_type
        self.pooling = pooling

        # Node feature encoder
        self.node_encoder = nn.Sequential(
            nn.Linear(node_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Edge feature encoder (optional, for edge-conditioned GNN)
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_feature_dim, hidden_dim // 2),
            nn.ReLU(),
        )

        # GNN layers
        self.gnn_layers = nn.ModuleList()
        for i in range(num_layers):
            in_dim = hidden_dim if i == 0 else hidden_dim
            out_dim = hidden_dim

            if gnn_type == 'gcn':
                self.gnn_layers.append(GCNConv(in_dim, out_dim))
            elif gnn_type == 'gat':
                self.gnn_layers.append(GATConv(in_dim, out_dim, heads=4, concat=False))
            else:
                raise ValueError(f"Unknown GNN type: {gnn_type}")

        # Batch normalization
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)
        ])

        # Attention pooling (if selected)
        if pooling == 'attention':
            self.attention_pool = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.Tanh(),
                nn.Linear(hidden_dim // 2, 1),
            )

        # Final projection to embedding dimension
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )

        self.dropout = nn.Dropout(dropout)

        logger.info(
            f"SceneGraphEmbedder initialized: "
            f"{num_layers} {gnn_type.upper()} layers, "
            f"{hidden_dim}d hidden, {embedding_dim}d output"
        )

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through GNN

        Args:
            node_features: Node features (num_nodes, node_feature_dim)
            edge_index: Edge connectivity (2, num_edges)
            edge_features: Edge features (num_edges, edge_feature_dim)
            batch: Batch assignment for each node (for batched graphs)

        Returns:
            Graph embeddings (batch_size, embedding_dim)
        """
        # Encode node features
        x = self.node_encoder(node_features)

        # GNN layers with residual connections
        for i, (gnn_layer, bn) in enumerate(zip(self.gnn_layers, self.batch_norms)):
            x_residual = x

            # Apply GNN layer
            x = gnn_layer(x, edge_index)

            # Batch normalization
            x = bn(x)

            # Residual connection (except first layer)
            if i > 0:
                x = x + x_residual

            # Activation and dropout
            x = F.relu(x)
            x = self.dropout(x)

        # Graph-level pooling
        if batch is None:
            # Single graph
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        if self.pooling == 'mean':
            graph_embedding = global_mean_pool(x, batch)
        elif self.pooling == 'max':
            graph_embedding = global_max_pool(x, batch)
        elif self.pooling == 'attention':
            graph_embedding = self._attention_pooling(x, batch)
        else:
            raise ValueError(f"Unknown pooling method: {self.pooling}")

        # Final projection
        output = self.output_projection(graph_embedding)

        return output

    def _attention_pooling(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """Attention-based graph pooling"""
        # Compute attention scores
        attention_scores = self.attention_pool(x)  # (num_nodes, 1)
        attention_weights = torch.softmax(attention_scores, dim=0)

        # Weighted sum
        weighted_features = x * attention_weights
        graph_embedding = global_mean_pool(weighted_features, batch)

        return graph_embedding

    def encode_scene_graph(self, scene_graph: SceneGraph) -> torch.Tensor:
        """
        Encode a single scene graph to embedding

        Args:
            scene_graph: SceneGraph object

        Returns:
            Embedding tensor (embedding_dim,)
        """
        pyg_data = self.scene_graph_to_pyg(scene_graph)

        with torch.no_grad():
            embedding = self.forward(
                pyg_data.x,
                pyg_data.edge_index,
                pyg_data.edge_attr if hasattr(pyg_data, 'edge_attr') else None,
            )

        return embedding.squeeze(0)

    def encode_scene_graphs_batch(self, scene_graphs: List[SceneGraph]) -> torch.Tensor:
        """
        Encode a batch of scene graphs

        Args:
            scene_graphs: List of SceneGraph objects

        Returns:
            Batch of embeddings (batch_size, embedding_dim)
        """
        pyg_data_list = [self.scene_graph_to_pyg(sg) for sg in scene_graphs]
        batch_data = Batch.from_data_list(pyg_data_list)

        with torch.no_grad():
            embeddings = self.forward(
                batch_data.x,
                batch_data.edge_index,
                batch_data.edge_attr if hasattr(batch_data, 'edge_attr') else None,
                batch_data.batch,
            )

        return embeddings

    def scene_graph_to_pyg(self, scene_graph: SceneGraph) -> Data:
        """
        Convert SceneGraph to PyTorch Geometric Data object

        Args:
            scene_graph: SceneGraph object

        Returns:
            PyG Data object with node features and edge index
        """
        num_nodes = len(scene_graph.nodes)

        if num_nodes == 0:
            # Empty graph - return dummy data
            return Data(
                x=torch.zeros((1, self.node_feature_dim)),
                edge_index=torch.zeros((2, 0), dtype=torch.long),
            )

        # Extract node features
        node_features = []
        for node in scene_graph.nodes:
            features = self._node_to_features(node)
            node_features.append(features)

        node_features = torch.stack(node_features)

        # Extract edge information
        edge_index = []
        edge_features = []

        for edge in scene_graph.edges:
            # Validate node IDs
            if edge.source_id < num_nodes and edge.target_id < num_nodes:
                edge_index.append([edge.source_id, edge.target_id])
                edge_features.append(self._edge_to_features(edge))

        if len(edge_index) > 0:
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            edge_features = torch.stack(edge_features)
        else:
            # No edges - use self-loops
            edge_index = torch.arange(num_nodes).unsqueeze(0).repeat(2, 1)
            edge_features = torch.zeros((num_nodes, self.edge_feature_dim))

        # Create PyG Data object
        data = Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_features,
        )

        return data

    def _node_to_features(self, node: SceneGraphNode) -> torch.Tensor:
        """Convert scene graph node to feature vector"""
        features = []

        # Category one-hot encoding (simplified)
        category_map = {
            'cube': 0, 'box': 1, 'ball': 2, 'cylinder': 3, 'capsule': 4,
            'robot_gripper': 5, 'robot_arm': 6, 'table': 7, 'object': 8
        }
        category_idx = category_map.get(node.category, 8)
        category_onehot = torch.zeros(9)
        category_onehot[category_idx] = 1.0
        features.append(category_onehot)

        # Bounding box features (normalized)
        bbox = torch.tensor(node.bbox, dtype=torch.float32) / 640.0  # Normalize by image size
        features.append(bbox)

        # Color features (if available)
        color_map = {
            'red': [1, 0, 0], 'green': [0, 1, 0], 'blue': [0, 0, 1],
            'yellow': [1, 1, 0], 'cyan': [0, 1, 1], 'magenta': [1, 0, 1],
            'white': [1, 1, 1], 'black': [0, 0, 0], 'unknown': [0.5, 0.5, 0.5]
        }
        color = node.attributes.get('color', 'unknown')
        color_vec = torch.tensor(color_map.get(color, [0.5, 0.5, 0.5]), dtype=torch.float32)
        features.append(color_vec)

        # Size features
        size_map = {'small': 0, 'medium': 1, 'large': 2}
        size_idx = size_map.get(node.attributes.get('size', 'medium'), 1)
        size_onehot = torch.zeros(3)
        size_onehot[size_idx] = 1.0
        features.append(size_onehot)

        # 3D position (if available)
        if node.position_3d is not None:
            pos_3d = torch.tensor(node.position_3d, dtype=torch.float32)
        else:
            pos_3d = torch.zeros(3)
        features.append(pos_3d)

        # Concatenate all features
        feature_vector = torch.cat(features)

        # Pad or truncate to node_feature_dim
        if feature_vector.size(0) < self.node_feature_dim:
            padding = torch.zeros(self.node_feature_dim - feature_vector.size(0))
            feature_vector = torch.cat([feature_vector, padding])
        elif feature_vector.size(0) > self.node_feature_dim:
            feature_vector = feature_vector[:self.node_feature_dim]

        return feature_vector

    def _edge_to_features(self, edge: SceneGraphEdge) -> torch.Tensor:
        """Convert scene graph edge to feature vector"""
        features = []

        # Relationship one-hot encoding
        relation_map = {
            'on': 0, 'above': 1, 'below': 2, 'left_of': 3, 'right_of': 4,
            'near': 5, 'far_from': 6, 'holding': 7, 'grasping': 8, 'touching': 9
        }
        relation_idx = relation_map.get(edge.relationship, 5)  # Default to 'near'
        relation_onehot = torch.zeros(10)
        relation_onehot[relation_idx] = 1.0
        features.append(relation_onehot)

        # Confidence score
        confidence = torch.tensor([edge.confidence], dtype=torch.float32)
        features.append(confidence)

        # Concatenate
        feature_vector = torch.cat(features)

        # Pad or truncate to edge_feature_dim
        if feature_vector.size(0) < self.edge_feature_dim:
            padding = torch.zeros(self.edge_feature_dim - feature_vector.size(0))
            feature_vector = torch.cat([feature_vector, padding])
        elif feature_vector.size(0) > self.edge_feature_dim:
            feature_vector = feature_vector[:self.edge_feature_dim]

        return feature_vector


class SimpleSceneGraphEmbedder(nn.Module):
    """
    Simplified scene graph embedder without torch_geometric

    Uses basic MLP for encoding when torch_geometric is not available.
    """

    def __init__(
        self,
        max_nodes: int = 10,
        node_feature_dim: int = 128,
        embedding_dim: int = 256,
        hidden_dim: int = 512,
    ):
        super().__init__()

        self.max_nodes = max_nodes
        self.node_feature_dim = node_feature_dim
        self.embedding_dim = embedding_dim

        # Simple MLP encoder
        input_dim = max_nodes * node_feature_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )

    def forward(self, node_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            node_features: (batch_size, max_nodes, node_feature_dim)

        Returns:
            Embeddings: (batch_size, embedding_dim)
        """
        batch_size = node_features.size(0)
        flattened = node_features.view(batch_size, -1)
        return self.encoder(flattened)


if __name__ == "__main__":
    # Test scene graph embedder
    from scene_graph_generator import SceneGraph, SceneGraphNode, SceneGraphEdge

    # Create dummy scene graph
    nodes = [
        SceneGraphNode(node_id=0, category='cube', attributes={'color': 'red', 'size': 'medium'}, bbox=(100, 100, 50, 50)),
        SceneGraphNode(node_id=1, category='ball', attributes={'color': 'blue', 'size': 'small'}, bbox=(200, 150, 40, 40)),
        SceneGraphNode(node_id=2, category='table', attributes={'color': 'white', 'size': 'large'}, bbox=(0, 400, 640, 80)),
    ]

    edges = [
        SceneGraphEdge(source_id=0, target_id=2, relationship='on', confidence=0.9),
        SceneGraphEdge(source_id=1, target_id=2, relationship='on', confidence=0.85),
        SceneGraphEdge(source_id=0, target_id=1, relationship='near', confidence=0.7),
    ]

    scene_graph = SceneGraph(nodes=nodes, edges=edges)

    # Test embedder
    if HAS_TORCH_GEOMETRIC:
        embedder = SceneGraphEmbedder(embedding_dim=256)
        embedder.eval()

        embedding = embedder.encode_scene_graph(scene_graph)
        print(f"Scene graph embedding shape: {embedding.shape}")
        print(f"Embedding norm: {embedding.norm().item():.3f}")

        # Test batch encoding
        embeddings = embedder.encode_scene_graphs_batch([scene_graph, scene_graph])
        print(f"Batch embeddings shape: {embeddings.shape}")
    else:
        print("torch_geometric not available, using SimpleSceneGraphEmbedder")
        embedder = SimpleSceneGraphEmbedder(embedding_dim=256)
        dummy_features = torch.randn(1, 10, 128)
        embedding = embedder(dummy_features)
        print(f"Embedding shape: {embedding.shape}")
