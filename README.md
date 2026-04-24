# Self-Pruning Neural Network

This project implements the Self-Pruning Neural Network case study for the Tredence Analytics AI Engineer internship. 

The core idea is simple but powerful: Instead of pruning a trained model in a separate step, attach a learnable gate scalar to every weight. Penalize the model during training for keeping too many gates open. The network learns what to keep—and what to throw away—on its own.

This repository contains a clean, standalone PyTorch script that handles custom layer creation, dataset loading, training routines, and automatic result visualization. It also includes an interactive real-time dashboard to monitor training!

## How It Works

### The Gated Weight Mechanism
Every `PrunableLinear` layer replaces the standard PyTorch linear layer with:
```python
gates         = torch.sigmoid(gate_scores)        # ∈ (0, 1) per weight
pruned_weight = weight * gates                    # element-wise masking
output        = F.linear(x, pruned_weight, bias)  # standard linear op
```
`gate_scores` is a learnable parameter tensor with the same shape as `weight`. When a gate collapses to 0, the corresponding weight is silenced—that connection is pruned without ever being removed from the graph. Gradients flow through both weight and `gate_scores` automatically.

### The Loss Function
```text
Total Loss = CrossEntropyLoss + λ × SparsityLoss 
SparsityLoss = mean(sigmoid(gate_scores))
```
- **`λ` (lambda)** controls the sparsity-accuracy trade-off.
- The **L1 norm** on gate values encourages exact zeros (unlike L2 which only shrinks).
- **Normalization by mean** keeps the loss scale stable across model sizes.
- *Note: The sparsity loss gradients are internally scaled by a balancing constant (5000) during backpropagation to prevent them from vanishing due to the mean normalization, allowing the Adam optimizer to effectively close gates.*

## Project Structure
```text
TREDENCE_AI/ 
├── self_pruning_network.py    ← Complete implementation in one standalone script
├── requirements.txt           ← Dependencies (torch, torchvision, matplotlib)
├── dashboard/                 ← Full-stack interactive real-time dashboard
│   ├── index.html
│   ├── styles.css
│   └── app.js
├── data/                      ← Auto-downloaded CIFAR-10 dataset
└── results/                   ← Auto-generated upon execution
    ├── metrics.json           ← Live telemetry for dashboard
    ├── training_curves.png
    ├── gate_distribution.png
    └── report.md
```

## Quick Start

### Prerequisites
- Python 3.8+ (64-bit recommended)

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Training Script
```bash
python self_pruning_network.py
```
*Note: The CIFAR-10 dataset will be automatically downloaded to `./data/` on the first run.*

### 3. Open the Interactive Dashboard
While the script is training, simply open `dashboard/index.html` in your web browser. 
You will see real-time, interactive charts displaying:
- Classification Loss vs. Validation Loss
- Validation Accuracy vs. Network Sparsity
- Real-time Layer-wise Sparsity
- Final Gate Distribution (available post-training)

## Insights & Trade-off Analysis

### The Pareto Frontier
As `λ` increases, sparsity increases but accuracy inevitably decays. The relationship traces a Pareto frontier—no single configuration dominates on both axes simultaneously. Finding the "sweet spot" yields significant sparsity with negligible accuracy loss, making it the best practical choice for deployment.

### Layer-wise Behavior
Not all layers prune equally. Early layers tend to retain denser connections because they extract low-level features (edges, textures) that are broadly reused. Deeper layers, which encode more task-specific patterns, prune faster under L1 pressure.
