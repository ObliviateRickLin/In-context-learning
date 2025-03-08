import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import yaml

# Add the src directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.tasks import get_task_sampler
from src.models import TransformerModel

def load_model(checkpoint_path):
    """Load a model from a checkpoint."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    
    # 加载状态字典
    state = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    
    # 从配置文件中获取模型配置
    config_path = os.path.join(os.path.dirname(checkpoint_path), "config.yaml")
    with open(config_path) as fp:
        conf = yaml.safe_load(fp)
    
    # 构建模型
    model = TransformerModel(
        n_dims=conf['model']['n_dims'],
        n_positions=conf['model']['n_positions'],
        n_embd=conf['model']['n_embd'],
        n_layer=conf['model']['n_layer'],
        n_head=conf['model']['n_head'],
    )
    
    # 加载模型状态
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    return model

def evaluate_model(model, task_name, n_examples, batch_size=64):
    """Evaluate the model on a task."""
    # Get the task
    task_kwargs = {}
    if task_name == "gaussian_kernel_regression":
        task_kwargs = {
            "R": 10,
            "sigma": 1.0,
            "scale": 1.0
        }
    
    task_sampler = get_task_sampler(task_name, n_dims=20, batch_size=batch_size, **task_kwargs)
    task = task_sampler()
    
    # Evaluate model for different numbers of in-context examples
    losses = []
    # 使用从 10 到 40 的上下文长度
    n_examples_list = list(range(10, 41, 2))  # 10, 12, 14, ..., 40
    
    for n in n_examples_list:
        # Generate data
        n_dims = 20
        xs = torch.randn(batch_size, n+1, n_dims)  # [batch_size, n_points+1, n_dims]
        ys = task.evaluate(xs)
        
        # Get model predictions
        with torch.no_grad():
            pred = model(xs, ys)
        
        # Calculate loss for the last point
        metric = task.get_metric()
        loss = metric(pred[:, -1], ys[:, -1]).mean().item()
        losses.append(loss)
    
    return n_examples_list, losses, task

def plot_loss(n_examples_list, losses, baseline_loss=None, title="Model Performance", output_dir="results"):
    """Plot the loss against the number of in-context examples."""
    plt.figure(figsize=(10, 6))
    plt.plot(n_examples_list, losses, marker='o', label="Model")
    
    if baseline_loss is not None:
        plt.axhline(y=baseline_loss, color='r', linestyle='--', label="Baseline")
    
    plt.xlabel("Number of In-Context Examples")
    plt.ylabel("Squared Error")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    
    # Save the figure
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "loss_plot.png"))
    plt.close()

def plot_kernel_function(model, output_dir="results"):
    """Plot the Gaussian kernel function."""
    # 创建一个高斯核回归任务
    task_kwargs = {
        "R": 10,
        "sigma": 1.0,
        "scale": 1.0
    }
    
    task_sampler = get_task_sampler("gaussian_kernel_regression", n_dims=20, batch_size=1, **task_kwargs)
    task = task_sampler()
    
    # 为了简化可视化，我们只关注前两个维度
    n_points = 100
    x_range = np.linspace(-3, 3, n_points)
    y_range = np.linspace(-3, 3, n_points)
    X, Y = np.meshgrid(x_range, y_range)
    
    # 计算每个中心的高斯核值
    centers = task.centers[0, :, :2].numpy()  # 只取第一个批次的中心，只关注前两个维度
    alphas = task.alphas[0, :].numpy()  # 只取第一个批次的权重
    sigma = task.sigma
    
    # 绘制每个中心的高斯核函数
    plt.figure(figsize=(12, 10))
    
    # 计算总的核函数值
    Z_total = np.zeros_like(X)
    
    for i, (center, alpha) in enumerate(zip(centers, alphas)):
        Z = np.zeros_like(X)
        for j in range(n_points):
            for k in range(n_points):
                x = np.array([X[j, k], Y[j, k]])
                dist = np.sum((x - center) ** 2)
                Z[j, k] = alpha * np.exp(-dist / (2 * sigma ** 2))
        
        # 绘制单个核函数
        plt.subplot(3, 4, i + 1)
        plt.contourf(X, Y, Z, levels=50, cmap='viridis')
        plt.colorbar(label=f'Kernel {i+1}')
        plt.title(f'Center: ({center[0]:.2f}, {center[1]:.2f})\nAlpha: {alpha:.2f}')
        
        # 累加到总核函数
        Z_total += Z
    
    # 保存单个核函数图
    os.makedirs(output_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "individual_kernels.png"))
    plt.close()
    
    # 绘制总的核函数
    plt.figure(figsize=(10, 8))
    plt.contourf(X, Y, Z_total, levels=50, cmap='viridis')
    plt.colorbar(label='Total Kernel Value')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Combined Gaussian Kernel Function')
    plt.savefig(os.path.join(output_dir, "combined_kernel.png"))
    plt.close()
    
    # 绘制3D表面
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z_total, cmap='viridis', edgecolor='none')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Kernel Value')
    ax.set_title('3D Combined Gaussian Kernel Function')
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='Kernel Value')
    plt.savefig(os.path.join(output_dir, "combined_kernel_3d.png"))
    plt.close()

def test_robustness(model, task_name, n_examples, batch_size=64, output_dir="results"):
    """Test the model's robustness to input scaling."""
    # Get the task
    task_kwargs = {}
    if task_name == "gaussian_kernel_regression":
        task_kwargs = {
            "R": 10,
            "sigma": 1.0,
            "scale": 1.0
        }
    
    task_sampler = get_task_sampler(task_name, n_dims=20, batch_size=batch_size, **task_kwargs)
    task = task_sampler()
    
    # Evaluate model for different numbers of in-context examples
    original_losses = []
    doubled_losses = []
    # 使用从 10 到 40 的上下文长度
    n_examples_list = list(range(10, 41, 2))  # 10, 12, 14, ..., 40
    
    for n in n_examples_list:
        # Generate data
        n_dims = 20
        xs = torch.randn(batch_size, n+1, n_dims)  # [batch_size, n_points+1, n_dims]
        ys = task.evaluate(xs)
        
        # Get model predictions on original inputs
        with torch.no_grad():
            pred_original = model(xs, ys)
        
        # Calculate loss for the last point
        metric = task.get_metric()
        loss_original = metric(pred_original[:, -1], ys[:, -1]).mean().item()
        original_losses.append(loss_original)
        
        # Double the inputs
        xs_doubled = xs * 2
        ys_doubled = task.evaluate(xs_doubled)
        
        # Get model predictions on doubled inputs
        with torch.no_grad():
            pred_doubled = model(xs_doubled, ys_doubled)
        
        # Calculate loss for the last point
        loss_doubled = metric(pred_doubled[:, -1], ys_doubled[:, -1]).mean().item()
        doubled_losses.append(loss_doubled)
    
    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(n_examples_list, original_losses, marker='o', label="Original Inputs")
    plt.plot(n_examples_list, doubled_losses, marker='x', label="Doubled Inputs")
    
    # Add baseline
    baseline_loss = metric(torch.zeros_like(ys[:, -1]), ys[:, -1]).mean().item()
    plt.axhline(y=baseline_loss, color='r', linestyle='--', label="Baseline")
    
    plt.xlabel("Number of In-Context Examples")
    plt.ylabel("Squared Error")
    plt.title("Model Robustness to Input Scaling")
    plt.legend()
    plt.grid(True)
    
    # Save the figure
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "robustness_test.png"))
    plt.close()
    
    return original_losses, doubled_losses

def main():
    parser = argparse.ArgumentParser(description="Evaluate a model and visualize results")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the model checkpoint")
    parser.add_argument("--task", type=str, default="gaussian_kernel_regression", help="Task to evaluate on")
    parser.add_argument("--n_examples", type=int, default=40, help="Maximum number of context examples")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for evaluation")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory to save results")
    parser.add_argument("--test_robustness", action="store_true", help="Test model robustness to input scaling")
    parser.add_argument("--plot_kernel", action="store_true", help="Plot the kernel function")
    
    args = parser.parse_args()
    
    print(f"Task name: {args.task}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    model = load_model(args.checkpoint)
    
    # Evaluate model
    n_examples_list, losses, task = evaluate_model(model, args.task, args.n_examples, args.batch_size)
    
    # Calculate baseline loss (prediction of zeros)
    n_dims = 20
    xs = torch.randn(args.batch_size, 1, n_dims)  # [batch_size, 1, n_dims]
    ys = task.evaluate(xs)
    metric = task.get_metric()
    baseline_loss = metric(torch.zeros_like(ys), ys).mean().item()
    
    # Plot loss
    plot_loss(n_examples_list, losses, baseline_loss, output_dir=args.output_dir)
    
    # Plot kernel function if requested
    if args.plot_kernel:
        plot_kernel_function(model, output_dir=args.output_dir)
    
    # Test robustness if requested
    if args.test_robustness:
        test_robustness(model, args.task, args.n_examples, args.batch_size, output_dir=args.output_dir)
    
    print(f"Evaluation complete. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()
