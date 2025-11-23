"""
Scene Graph Visualization Tool

Interactive visualization of scene graphs overlaid on robot camera views.
Debugging tool for evaluating scene graph quality and relationships.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_agg import FigureCanvasAgg
import networkx as nx
from typing import Optional, Tuple
import logging

try:
    from vision.scene_graph_generator import SceneGraph
except ImportError:
    try:
        from ..vision.scene_graph_generator import SceneGraph
    except ImportError:
        from scene_graph_generator import SceneGraph

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SceneGraphVisualizer:
    """
    Visualize scene graphs on robot camera images

    Provides tools for debugging and analyzing scene graph quality.
    """

    def __init__(self, figsize: Tuple[int, int] = (15, 8)):
        self.figsize = figsize

    def visualize_on_image(
        self,
        rgb_image: np.ndarray,
        scene_graph: SceneGraph,
        show_boxes: bool = True,
        show_labels: bool = True,
        show_relationships: bool = True,
    ) -> np.ndarray:
        """
        Overlay scene graph on RGB image

        Args:
            rgb_image: RGB image (H, W, 3)
            scene_graph: SceneGraph object
            show_boxes: Show bounding boxes
            show_labels: Show object labels
            show_relationships: Show relationship arrows

        Returns:
            Visualization image
        """
        vis_image = rgb_image.copy()

        # Generate colors for nodes
        np.random.seed(42)
        colors = plt.cm.tab20(np.linspace(0, 1, len(scene_graph.nodes)))

        # Draw relationships first (as arrows)
        if show_relationships:
            for edge in scene_graph.edges:
                src_node = scene_graph.nodes[edge.source_id]
                tgt_node = scene_graph.nodes[edge.target_id]

                # Compute centers
                x1, y1, w1, h1 = src_node.bbox
                x2, y2, w2, h2 = tgt_node.bbox
                c1 = (int(x1 + w1/2), int(y1 + h1/2))
                c2 = (int(x2 + w2/2), int(y2 + h2/2))

                # Draw arrow
                color = (255, 255, 0) if edge.confidence > 0.7 else (128, 128, 128)
                cv2.arrowedLine(vis_image, c1, c2, color, 2, tipLength=0.3)

                # Add relationship label
                mid_x, mid_y = (c1[0] + c2[0]) // 2, (c1[1] + c2[1]) // 2
                cv2.putText(
                    vis_image,
                    edge.relationship,
                    (mid_x, mid_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255, 255, 255),
                    1
                )

        # Draw bounding boxes and labels
        for idx, node in enumerate(scene_graph.nodes):
            color = tuple((np.array(colors[idx][:3]) * 255).astype(int).tolist())

            if show_boxes:
                x, y, w, h = node.bbox
                cv2.rectangle(vis_image, (x, y), (x+w, y+h), color, 2)

            if show_labels:
                x, y, _, _ = node.bbox
                label_text = f"{node.category}"
                if 'color' in node.attributes:
                    label_text += f" ({node.attributes['color']})"

                # Background for text
                (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(vis_image, (x, y-text_h-5), (x+text_w, y), color, -1)
                cv2.putText(vis_image, label_text, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        return vis_image

    def visualize_graph_structure(self, scene_graph: SceneGraph) -> plt.Figure:
        """
        Visualize scene graph as network diagram

        Args:
            scene_graph: SceneGraph object

        Returns:
            Matplotlib figure
        """
        # Create networkx graph
        G = nx.DiGraph()

        # Add nodes
        for node in scene_graph.nodes:
            label = f"{node.category}"
            if 'color' in node.attributes:
                label += f"\n({node.attributes['color']})"
            G.add_node(node.node_id, label=label, category=node.category)

        # Add edges
        for edge in scene_graph.edges:
            G.add_edge(
                edge.source_id,
                edge.target_id,
                label=edge.relationship,
                weight=edge.confidence
            )

        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize)

        # Layout
        pos = nx.spring_layout(G, k=2, iterations=50)

        # Draw nodes
        node_colors = ['lightblue' if G.nodes[n]['category'] in ['cube', 'box', 'ball'] else 'lightgreen'
                       for n in G.nodes()]
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=2000, ax=ax)

        # Draw edges
        nx.draw_networkx_edges(G, pos, width=2, alpha=0.6, arrows=True, arrowsize=20, ax=ax)

        # Draw labels
        node_labels = {n: G.nodes[n]['label'] for n in G.nodes()}
        nx.draw_networkx_labels(G, pos, node_labels, font_size=10, ax=ax)

        edge_labels = {(u, v): G[u][v]['label'] for u, v in G.edges()}
        nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=8, ax=ax)

        ax.set_title("Scene Graph Structure", fontsize=14, fontweight='bold')
        ax.axis('off')

        plt.tight_layout()
        return fig

    def visualize_complete(
        self,
        rgb_image: np.ndarray,
        scene_graph: SceneGraph,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Complete visualization with image and graph structure

        Args:
            rgb_image: RGB image
            scene_graph: SceneGraph object
            save_path: Optional path to save figure

        Returns:
            Matplotlib figure
        """
        fig = plt.figure(figsize=(20, 8))

        # Left: Image with overlays
        ax1 = plt.subplot(1, 2, 1)
        vis_image = self.visualize_on_image(rgb_image, scene_graph)
        ax1.imshow(cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB))
        ax1.set_title("Scene with Detected Objects and Relations", fontsize=12, fontweight='bold')
        ax1.axis('off')

        # Right: Graph structure
        ax2 = plt.subplot(1, 2, 2)
        G = nx.DiGraph()
        for node in scene_graph.nodes:
            G.add_node(node.node_id, label=node.category)
        for edge in scene_graph.edges:
            G.add_edge(edge.source_id, edge.target_id, label=edge.relationship)

        pos = nx.spring_layout(G, k=2)
        nx.draw_networkx_nodes(G, pos, node_size=1500, node_color='lightblue', ax=ax2)
        nx.draw_networkx_edges(G, pos, width=2, alpha=0.6, arrows=True, arrowsize=15, ax=ax2)
        nx.draw_networkx_labels(G, pos, {n: G.nodes[n]['label'] for n in G.nodes()}, font_size=9, ax=ax2)
        nx.draw_networkx_edge_labels(G, pos, {(u, v): G[u][v]['label'] for u, v in G.edges()}, font_size=7, ax=ax2)

        ax2.set_title("Scene Graph", fontsize=12, fontweight='bold')
        ax2.axis('off')

        plt.suptitle(f"Scene Graph Visualization ({len(scene_graph.nodes)} objects, {len(scene_graph.edges)} relations)",
                     fontsize=14, fontweight='bold')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Visualization saved to {save_path}")

        return fig


if __name__ == "__main__":
    logger.info("Scene graph visualizer module loaded")
