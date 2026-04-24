import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np
import os
import json

class PrunableLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(PrunableLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        
        # gate_scores parameter with the exact same shape as weight
        self.gate_scores = nn.Parameter(torch.Tensor(out_features, in_features))
        
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / np.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
            
        # Initialize gate scores to -1.0 so sigmoid starts at ~0.27
        nn.init.constant_(self.gate_scores, -1.0)

    def forward(self, x):
        gates = torch.sigmoid(self.gate_scores)
        pruned_weights = self.weight * gates
        return F.linear(x, pruned_weights, self.bias)

    def get_gates(self):
        return torch.sigmoid(self.gate_scores)
        
    def sparsity_stats(self, threshold=1e-2):
        gates = self.get_gates()
        pruned = torch.sum(gates < threshold).item()
        total = gates.numel()
        return pruned / total

class SelfPruningNetwork(nn.Module):
    def __init__(self):
        super(SelfPruningNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = PrunableLinear(3072, 512)
        self.relu1 = nn.ReLU()
        self.fc2 = PrunableLinear(512, 256)
        self.relu2 = nn.ReLU()
        self.fc3 = PrunableLinear(256, 128)
        self.relu3 = nn.ReLU()
        self.fc4 = PrunableLinear(128, 10)
        
    def forward(self, x):
        x = self.flatten(x)
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.relu3(self.fc3(x))
        x = self.fc4(x)
        return x
        
    def get_all_prunable_layers(self):
        return [self.fc1, self.fc2, self.fc3, self.fc4]

    def network_sparsity_loss(self):
        """Computes the mean gate value across the entire network."""
        total_sum = 0.0
        total_elements = 0
        for layer in self.get_all_prunable_layers():
            gates = layer.get_gates()
            total_sum += torch.sum(gates)
            total_elements += gates.numel()
        return total_sum / total_elements

def get_dataloaders(batch_size=128):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    
    full_trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    
    # Validation split
    train_size = int(0.9 * len(full_trainset))
    val_size = len(full_trainset) - train_size
    trainset, valset = random_split(full_trainset, [train_size, val_size])
    
    # Note: valset inherits transform_train. Ideally we'd set transform_test, 
    # but for simplicity in this script we keep it as is, or use transform_test.
    valset.dataset.transform = transform_test
    
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, val_loader, test_loader

def train_one_epoch(model, train_loader, optimizer, criterion, lambda_reg, device):
    model.train()
    running_loss, running_cls, running_sp = 0.0, 0.0, 0.0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        
        outputs = model(inputs)
        cls_loss = criterion(outputs, labels)
        sp_loss = model.network_sparsity_loss()
        
        total_loss = cls_loss + lambda_reg * sp_loss
        
        # Check for NaN/Inf anomalies
        if not torch.isfinite(total_loss):
            print("WARNING: non-finite loss, ending epoch early")
            break
            
        total_loss.backward()
        
        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        running_loss += total_loss.item()
        running_cls += cls_loss.item()
        running_sp += sp_loss.item()
        
    return running_loss / len(train_loader), running_cls / len(train_loader), running_sp / len(train_loader)

def evaluate(model, data_loader, criterion, lambda_reg, device, threshold=1e-2):
    model.eval()
    correct, total, running_loss = 0, 0, 0.0
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    accuracy = 100 * correct / total
    avg_loss = running_loss / len(data_loader)
    
    # Sparsity
    pruned_count = 0
    total_count = 0
    gate_values = []
    layer_sparsity = []
    
    with torch.no_grad():
        for layer in model.get_all_prunable_layers():
            gates = layer.get_gates()
            gate_values.extend(gates.cpu().numpy().flatten())
            layer_pruned = torch.sum(gates < threshold).item()
            layer_total = gates.numel()
            pruned_count += layer_pruned
            total_count += layer_total
            layer_sparsity.append(layer_pruned / layer_total * 100)
            
    overall_sparsity = 100 * pruned_count / total_count
    
    return avg_loss, accuracy, overall_sparsity, layer_sparsity, np.array(gate_values)

def plot_training_curves(metrics_dict, output_dir):
    lambdas = list(metrics_dict.keys())
    epochs = range(1, len(metrics_dict[lambdas[0]]['train_loss']) + 1)
    
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Training Classification Loss
    plt.subplot(1, 2, 1)
    for lam in lambdas:
        plt.plot(epochs, metrics_dict[lam]['train_cls_loss'], label=f'λ={lam}')
    plt.title('Training Classification Loss vs Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot 2: Validation Accuracy
    plt.subplot(1, 2, 2)
    for lam in lambdas:
        plt.plot(epochs, metrics_dict[lam]['val_acc'], label=f'λ={lam}')
    plt.title('Validation Accuracy vs Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'))
    plt.close()

def plot_gate_distribution(gate_values, lambda_val, output_dir):
    plt.figure(figsize=(10, 6))
    plt.hist(gate_values, bins=50, range=(0, 1), alpha=0.75, color='teal', edgecolor='black')
    plt.title(f'Distribution of Gate Values (Lambda = {lambda_val})', fontsize=14)
    plt.xlabel('Gate Value (Sigmoid Output)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(axis='y', alpha=0.5)
    plt.savefig(os.path.join(output_dir, 'gate_distribution.png'))
    plt.close()

def generate_report(results, best_lambda, output_dir):
    report_path = os.path.join(output_dir, 'report.md')
    with open(report_path, 'w') as f:
        f.write("# Self-Pruning Network Results\n\n")
        f.write("## Experiment Summary\n")
        f.write("| Lambda | Test Accuracy | Sparsity (%) |\n")
        f.write("|--------|---------------|--------------|\n")
        for lam, res in results.items():
            f.write(f"| {lam:<6} | {res['test_acc']:>12.2f}% | {res['test_sparsity']:>11.2f}% |\n")
        
        f.write(f"\n**Best Model Configuration:** Lambda = {best_lambda}\n\n")
        f.write("## Visualizations\n")
        f.write("- [Training Curves](./training_curves.png)\n")
        f.write("- [Gate Distribution](./gate_distribution.png)\n")

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    os.makedirs('./results', exist_ok=True)
    
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=128)
    
    lambdas = [0.01, 0.1, 0.5]
    num_epochs = 25
    
    results = {}
    metrics_log = {}
    
    best_score = -1
    best_lambda = None
    best_gates = None
    
    for lam in lambdas:
        print(f"\n{'='*40}\nTraining with lambda = {lam}\n{'='*40}")
        model = SelfPruningNetwork().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
        
        metrics_log[str(lam)] = {
            'train_loss': [], 'train_cls_loss': [], 'train_sp_loss': [],
            'val_loss': [], 'val_acc': [], 'val_sparsity': [], 'layer_sparsity': []
        }
        
        for epoch in range(num_epochs):
            t_loss, t_cls, t_sp = train_one_epoch(model, train_loader, optimizer, criterion, lam, device)
            v_loss, v_acc, v_sp, l_sp, _ = evaluate(model, val_loader, criterion, lam, device)
            
            scheduler.step()
            
            metrics_log[str(lam)]['train_loss'].append(t_loss)
            metrics_log[str(lam)]['train_cls_loss'].append(t_cls)
            metrics_log[str(lam)]['train_sp_loss'].append(t_sp)
            metrics_log[str(lam)]['val_loss'].append(v_loss)
            metrics_log[str(lam)]['val_acc'].append(v_acc)
            metrics_log[str(lam)]['val_sparsity'].append(v_sp)
            metrics_log[str(lam)]['layer_sparsity'].append(l_sp)
            
            print(f"Epoch [{epoch+1:02d}/{num_epochs}] "
                  f"TrLoss: {t_loss:.4f} | ValAcc: {v_acc:.2f}% | Sparsity: {v_sp:.2f}%")
            
            # Save metrics live
            with open('./results/metrics.json', 'w') as f:
                json.dump(metrics_log, f)
                
        # Final test evaluation
        test_loss, test_acc, test_sp, _, gate_values = evaluate(model, test_loader, criterion, lam, device)
        print(f"--> Final Test (lambda={lam}): Acc = {test_acc:.2f}%, Sparsity = {test_sp:.2f}%")
        results[lam] = {'test_acc': test_acc, 'test_sparsity': test_sp}
        
        score = test_acc + (test_sp * 0.5)
        if score > best_score:
            best_score = score
            best_lambda = lam
            best_gates = gate_values
            
    plot_training_curves(metrics_log, './results')
    if best_gates is not None:
        plot_gate_distribution(best_gates, best_lambda, './results')
    
    generate_report(results, best_lambda, './results')
    print("\nExperiment complete. Results saved to ./results/")

if __name__ == '__main__':
    torch.manual_seed(42)
    np.random.seed(42)
    main()