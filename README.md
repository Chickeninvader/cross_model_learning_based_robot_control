# Cross-Embodiment Robotic Learning with Scene Graphs

Cross-embodiment transfer learning between Franka and Sawyer robots using panoptic scene graphs as morphology-agnostic representations.

## Quick Start

### Installation

```bash
# Clone with submodules
git clone --recursive https://github.com/ChefToan/Cross-Model-Learning-Robot-Control.git
cd Cross-Model-Learning-Robot-Control

# Install dependencies
pip install -r requirements.txt

# Initialize submodules
git submodule update --init --recursive
```

### Environment Creation and Simulation

```bash
# Create custom environment with objects
python src/create_env.py --num-objects 5 --type box ball --colors random --save my_env

# Run teleoperation with SpaceMouse
python src/run_simulation.py --mode teleop --env my_env

# Record demonstrations
python src/run_simulation.py --mode teleop --env my_env --record --num-episodes 5

# Replay recorded demonstrations
python src/replay_demo.py data/recordings/my_env_*/
```

## Project Structure

```
├── src/
│   ├── vision/          # Scene graph pipeline (SAM, VLPrompt, GNN)
│   ├── policy/          # Transformer policy & behavior cloning
│   ├── training/        # Cross-embodiment training
│   ├── evaluation/      # Zero-shot transfer evaluation
│   ├── utils/           # Data loading & visualization
│   ├── create_env.py    # Environment creation
│   ├── run_simulation.py # Teleoperation and recording
│   ├── replay_demo.py   # Demonstration replay
│   └── custom_spacemouse.py # SpaceMouse driver
├── configs/             # YAML configuration files
├── scripts/             # Training scripts for Sol supercomputer
├── external/            # Git submodules (SAM, VLPrompt, Pix2Grp, etc.)
└── data/                # Datasets and recordings
```

## Key Components

### Scene Graph Pipeline
- **SAM Segmentation**: Object detection with <100ms inference
- **VLPrompt**: Language-guided scene graph generation (80%+ accuracy)
- **GNN Embedder**: Graph neural network encoding (256d embeddings)

### Cross-Embodiment Policy
- **Transformer Architecture**: 4 layers, 8 attention heads, 512d hidden
- **Scene Graph Conditioning**: Cross-attention mechanism
- **Robot-Specific Heads**: Separate action decoders for Franka/Sawyer

### Training & Evaluation
- **Behavior Cloning**: Curriculum learning with data augmentation
- **Mixed-Robot Training**: Joint training on Franka + Sawyer demonstrations
- **Ablation Tools**: Automated comparison of model variants

## External Research Repositories

This project integrates multiple state-of-the-art models for scene understanding and robot learning:

### Scene Graph Generation
- **VLPrompt** - Vision-language scene graph generation with panoptic segmentation
- **Pix2Grp** - Pixel-to-graph scene understanding (CVPR 2024)
- **Scene-Graph-Benchmark** - PyTorch implementation of scene graph detection
- **Graph R-CNN** - Object detection with scene graph generation
- **fair-psgg** - Panoptic scene graph generation

### Segmentation Models
- **Segment Anything (SAM)** - Foundation model for object segmentation
- **Lang-Segment-Anything** - Language-guided SAM with text prompts

### Cross-Embodiment Learning
- **CrossFormer** - Cross-embodiment transformer (900K+ trajectories)
- **Octo** - Transformer diffusion policy (800K+ trajectories)
- **RoboMimic** - Imitation learning framework

### Training Scripts

Training scripts for Sol supercomputer are available in `scripts/`:
- `train_vlprompt_sol.sh` - VLPrompt scene graph generation
- `train_sam_sol.sh` - SAM segmentation
- `train_pix2grp_sol.sh` - Pix2Grp scene understanding
- `train_psgg_sol.sh` - Panoptic scene graph generation
- `train_scene_graph_benchmark_sol.sh` - Scene graph detection
- `train_crossformer_sol.sh` - Cross-embodiment transformer
- `train_octo_sol.sh` - Octo diffusion policy
- `train_robomimic_sol.sh` - RoboMimic behavior cloning

## Additional Resources

- **Product Backlogs**: Sprint documentation available in `product-backlogs/`
- **Environment & Simulation**: Custom environment creation and teleoperation tools in `src/`

## Citation

```bibtex
@misc{pham2025crossembodiment,
  title={AI-Driven Robotic Manipulation: Cross-Embodiment Learning with Panoptic Scene Graphs},
  author={Pham, Toan and Vo, Albert},
  year={2025},
  institution={Arizona State University}
}
```

## License

See LICENSE file for details.

## Contact

- Team: Toan Pham, Albert Vo
- Sponsor: Dr. Nakul Gopalan
- Repository: https://github.com/ChefToan/Cross-Model-Learning-Robot-Control
