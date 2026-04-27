"""
Problem 4: 网络优化器对比实验
2维双曲损失函数: L(θ1, θ2) = θ1^2 - θ2^2
鞍点在 (0, 0)
对比 SGD、RMSprop、AdaDelta、Adam 四种优化器
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D


def hyperbolic_loss(theta1, theta2):
    """双曲损失函数: L = θ1^2 - θ2^2"""
    return theta1**2 - theta2**2


def gradient_hyperbolic(theta1, theta2):
    """双曲损失函数的梯度"""
    dL_dtheta1 = 2 * theta1
    dL_dtheta2 = -2 * theta2
    return np.array([dL_dtheta1, dL_dtheta2])


class OptimizerBase:
    """优化器基类"""
    def __init__(self, lr=0.1):
        self.lr = lr
        self.theta = None
        self.history = []

    def initialize(self, theta0):
        self.theta = np.array(theta0, dtype=np.float64)
        self.history = [self.theta.copy()]

    def step(self, grad_fn):
        raise NotImplementedError

    def optimize(self, grad_fn, num_steps=100):
        for _ in range(num_steps):
            grad = grad_fn(self.theta[0], self.theta[1])
            self.step(grad)
            self.history.append(self.theta.copy())
        return np.array(self.history)


class SGD(OptimizerBase):
    """SGD优化器: θ^(k+1) = θ^(k) - α * g^(k)"""
    def step(self, grad):
        self.theta = self.theta - self.lr * grad


class RMSprop(OptimizerBase):
    """
    RMSprop优化器
    v^(k) = β * v^(k-1) + (1-β) * (g^(k))^2
    θ^(k+1) = θ^(k) - α * g^(k) / sqrt(v^(k) + ε)
    """
    def __init__(self, lr=0.1, beta=0.9, eps=1e-8):
        super().__init__(lr)
        self.beta = beta
        self.eps = eps
        self.v = None

    def initialize(self, theta0):
        super().initialize(theta0)
        self.v = np.zeros(2)

    def step(self, grad):
        self.v = self.beta * self.v + (1 - self.beta) * (grad ** 2)
        self.theta = self.theta - self.lr * grad / (np.sqrt(self.v) + self.eps)


class AdaDelta(OptimizerBase):
    """
    AdaDelta优化器 (公式 7.29)
    v^(k) = β * v^(k-1) + (1-β) * (g^(k))^2
    Δθ^(k) = -sqrt(u^(k-1) + ε) / sqrt(v^(k) + ε) * g^(k)
    u^(k) = β * u^(k-1) + (1-β) * (Δθ^(k))^2
    θ^(k+1) = θ^(k) + Δθ^(k)
    """
    def __init__(self, beta=0.9, eps=1e-8):
        super().__init__(lr=1.0)  # AdaDelta的学习率自适应，lr参数不使用
        self.beta = beta
        self.eps = eps
        self.v = None
        self.u = None

    def initialize(self, theta0):
        super().initialize(theta0)
        self.v = np.zeros(2)
        self.u = np.zeros(2)

    def step(self, grad):
        self.v = self.beta * self.v + (1 - self.beta) * (grad ** 2)
        delta_theta = -np.sqrt(self.u + self.eps) / np.sqrt(self.v + self.eps) * grad
        self.u = self.beta * self.u + (1 - self.beta) * (delta_theta ** 2)
        self.theta = self.theta + delta_theta


class Adam(OptimizerBase):
    """
    Adam优化器
    m^(k) = β1 * m^(k-1) + (1-β1) * g^(k)
    v^(k) = β2 * v^(k-1) + (1-β2) * (g^(k))^2
    m_hat = m^(k) / (1 - β1^k)
    v_hat = v^(k) / (1 - β2^k)
    θ^(k+1) = θ^(k) - α * m_hat / (sqrt(v_hat) + ε)
    """
    def __init__(self, lr=0.1, beta1=0.9, beta2=0.999, eps=1e-8):
        super().__init__(lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = None
        self.v = None
        self.t = 0

    def initialize(self, theta0):
        super().initialize(theta0)
        self.m = np.zeros(2)
        self.v = np.zeros(2)
        self.t = 0

    def step(self, grad):
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grad ** 2)
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)
        self.theta = self.theta - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


def visualize_optimization_2d(optimizers_results, theta_range=2.0, num_steps=100):
    """
    2D可视化优化过程在损失景观上的轨迹
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    # 创建网格用于绘制等高线
    theta1_grid = np.linspace(-theta_range, theta_range, 400)
    theta2_grid = np.linspace(-theta_range, theta_range, 400)
    Theta1, Theta2 = np.meshgrid(theta1_grid, theta2_grid)
    Loss = hyperbolic_loss(Theta1, Theta2)

    colors = ['blue', 'green', 'orange', 'purple']

    for idx, (name, trajectory) in enumerate(optimizers_results.items()):
        ax = axes[idx]

        # 绘制等高线
        levels = np.linspace(-3, 3, 21)
        contour = ax.contour(Theta1, Theta2, Loss, levels=levels, cmap='coolwarm', alpha=0.6)
        ax.clabel(contour, inline=True, fontsize=8)

        # 绘制鞍点
        ax.scatter([0], [0], color='red', s=200, marker='*', zorder=10, label='Saddle Point (0,0)')

        # 绘制优化轨迹
        traj = trajectory[:num_steps+1]
        ax.plot(traj[:, 0], traj[:, 1], 'k-', linewidth=1.5, alpha=0.5, label='Trajectory')
        ax.scatter(traj[0, 0], traj[0, 1], color='green', s=100, marker='o', zorder=10, label='Start')
        ax.scatter(traj[-1, 0], traj[-1, 1], color='red', s=100, marker='x', zorder=10, label='End')

        # 添加箭头表示方向
        for i in range(0, len(traj)-1, max(1, len(traj)//20)):
            dx = traj[i+1, 0] - traj[i, 0]
            dy = traj[i+1, 1] - traj[i, 1]
            ax.annotate('', xy=(traj[i+1, 0], traj[i+1, 1]), xytext=(traj[i, 0], traj[i, 1]),
                       arrowprops=dict(arrowstyle='->', color=colors[idx], lw=1.5, alpha=0.7))

        # 添加步数标签
        for i in range(0, len(traj), max(1, len(traj)//5)):
            ax.annotate(f'{i}', (traj[i, 0], traj[i, 1]), fontsize=8, alpha=0.7)

        ax.set_xlabel('θ₁')
        ax.set_ylabel('θ₂')
        ax.set_title(f'{name} Optimization Path')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-theta_range, theta_range)
        ax.set_ylim(-theta_range, theta_range)
        ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig('problem4_optimization_2d.png', dpi=150)
    print("2D可视化结果已保存到 problem4_optimization_2d.png")


def visualize_loss_landscape_3d():
    """3D可视化损失景观"""
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    theta1 = np.linspace(-2, 2, 100)
    theta2 = np.linspace(-2, 2, 100)
    Theta1, Theta2 = np.meshgrid(theta1, theta2)
    Loss = hyperbolic_loss(Theta1, Theta2)

    surf = ax.plot_surface(Theta1, Theta2, Loss, cmap='coolwarm', alpha=0.8, edgecolor='none')
    ax.scatter([0], [0], [0], color='black', s=200, marker='*', label='Saddle Point')
    ax.set_xlabel('θ₁')
    ax.set_ylabel('θ₂')
    ax.set_zlabel('L(θ₁, θ₂)')
    ax.set_title('Hyperbolic Loss Landscape: L = θ₁² - θ₂²')
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
    plt.savefig('problem4_loss_landscape_3d.png', dpi=150)
    print("3D损失景观已保存到 problem4_loss_landscape_3d.png")


def analyze_escape_from_saddle(optimizers_results, threshold=0.5):
    """
    分析各优化器逃离鞍点的速度
    定义: 当 ||θ|| > threshold 时认为已逃离鞍点
    """
    print("\n" + "="*60)
    print("优化器逃离鞍点速度分析")
    print("="*60)
    print(f"逃离阈值: ||θ|| > {threshold}")
    print("-"*60)

    escape_steps = {}

    for name, trajectory in optimizers_results.items():
        escape_step = None
        for i, theta in enumerate(trajectory):
            if np.linalg.norm(theta) > threshold:
                escape_step = i
                break
        escape_steps[name] = escape_step

        if escape_step is not None:
            print(f"{name:10s}: 第 {escape_step:3d} 步逃离鞍点")
        else:
            print(f"{name:10s}: 未在轨迹长度内逃离鞍点")

    # 可视化对比
    fig, ax = plt.subplots(figsize=(10, 6))

    names = list(escape_steps.keys())
    steps = [escape_steps[n] if escape_steps[n] is not None else 100 for n in names]
    colors = ['blue', 'green', 'orange', 'purple']

    bars = ax.bar(names, steps, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Steps to Escape Saddle Point')
    ax.set_title(f'Escape Speed Comparison (threshold ||θ|| > {threshold})')
    ax.grid(True, axis='y', alpha=0.3)

    # 添加数值标签
    for bar, step in zip(bars, steps):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{step}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig('problem4_escape_analysis.png', dpi=150)
    print("\n逃离速度分析已保存到 problem4_escape_analysis.png")

    return escape_steps


def analyze_trajectory_properties(optimizers_results):
    """分析轨迹的收敛性质"""
    print("\n" + "="*60)
    print("轨迹收敛性质分析")
    print("="*60)

    for name, trajectory in optimizers_results.items():
        # 计算最终位置和损失
        final_theta = trajectory[-1]
        final_loss = hyperbolic_loss(final_theta[0], final_theta[1])

        # 计算路径长度
        path_length = 0
        for i in range(len(trajectory) - 1):
            path_length += np.linalg.norm(trajectory[i+1] - trajectory[i])

        # 计算从起点到终点的直线距离
        direct_distance = np.linalg.norm(trajectory[-1] - trajectory[0])

        print(f"\n{name}:")
        print(f"  最终位置: θ=({final_theta[0]:.4f}, {final_theta[1]:.4f})")
        print(f"  最终损失: L={final_loss:.4f}")
        print(f"  路径长度: {path_length:.4f}")
        print(f"  直线距离: {direct_distance:.4f}")
        print(f"  路径/直线比: {path_length/direct_distance:.2f}")


def run_experiments():
    """运行所有优化器对比实验"""
    print("="*60)
    print("Problem 4: 网络优化器对比实验")
    print("="*60)
    print("损失函数: L(theta1, theta2) = theta1^2 - theta2^2")
    print("鞍点位置: (0, 0)")
    print("="*60)

    # 参数设置
    lr = 0.1  # 学习率（SGD、RMSprop、Adam使用）
    num_steps = 100
    theta0 = [0.5, 0.5]  # 初始点，靠近鞍点

    print(f"\n实验参数:")
    print(f"  学习率 alpha: {lr}")
    print(f"  迭代次数: {num_steps}")
    print(f"  初始点: theta0={theta0}")

    # 创建优化器
    optimizers = {
        'SGD': SGD(lr=lr),
        'RMSprop': RMSprop(lr=lr),
        'AdaDelta': AdaDelta(),
        'Adam': Adam(lr=lr),
    }

    # 运行优化
    results = {}
    for name, opt in optimizers.items():
        opt.initialize(theta0)
        trajectory = opt.optimize(gradient_hyperbolic, num_steps)
        results[name] = trajectory
        print(f"\n{name} 完成: 从 {theta0} 到 [{trajectory[-1][0]:.4f}, {trajectory[-1][1]:.4f}]")

    # 可视化
    visualize_optimization_2d(results, theta_range=2.0, num_steps=num_steps)
    visualize_loss_landscape_3d()

    # 分析逃离鞍点速度
    escape_steps = analyze_escape_from_saddle(results, threshold=0.8)

    # 分析收敛性质
    analyze_trajectory_properties(results)

    # 保存结果
    np.save('problem4_results.npy', results, allow_pickle=True)
    print("\n结果已保存到 problem4_results.npy")

    return results, escape_steps


def main():
    results, escape_steps = run_experiments()

    print("\n" + "="*60)
    print("实验完成!")
    print("="*60)

    return results


if __name__ == '__main__':
    main()
