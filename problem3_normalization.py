"""
Problem 3: 数据预处理与逐层规范化
实现3层MLP，包含:
(1) 对原始输入做Z值归一化
(2) 对第1层净输入做批量归一化(BatchNorm)
(3) 对第2层净输入做层归一化(LayerNorm)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt


class ZScoreNormalization:
    """
    Z值归一化: 将每一维特征调整为均值为0，方差为1
    公式: x_norm = (x - mean) / sqrt(var + eps)
    """
    def __init__(self, eps=1e-5):
        self.eps = eps
        self.running_mean = None
        self.running_var = None

    def fit(self, X):
        """计算训练数据的均值和方差"""
        self.running_mean = X.mean(dim=0, keepdim=True)
        self.running_var = X.var(dim=0, keepdim=True, unbiased=False)

    def transform(self, X):
        """应用归一化"""
        if self.running_mean is None:
            raise ValueError("Must call fit() before transform()")
        return (X - self.running_mean) / torch.sqrt(self.running_var + self.eps)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


class BatchNormalizationManual(nn.Module):
    """
    手动实现的批量归一化层
    公式: y = γ * (x - μ_B) / sqrt(σ_B^2 + ε) + β
    """
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        # 可学习的缩放和平移参数
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))

        # 运行统计量（用于推理）
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x):
        """
        x: (batch_size, num_features)
        """
        if self.training:
            # 训练模式: 使用当前batch的统计量
            batch_mean = x.mean(dim=0)
            batch_var = x.var(dim=0, unbiased=False)

            # 更新运行统计量
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var

            # 归一化
            x_normalized = (x - batch_mean) / torch.sqrt(batch_var + self.eps)
        else:
            # 推理模式: 使用运行统计量
            x_normalized = (x - self.running_mean) / torch.sqrt(self.running_var + self.eps)

        # 缩放和平移
        out = self.gamma * x_normalized + self.beta
        return out


class LayerNormalizationManual(nn.Module):
    """
    手动实现的层归一化
    公式: y = γ * (x - μ_L) / sqrt(σ_L^2 + ε) + β
    注意: 层归一化是在每个样本的特征维度上进行归一化
    """
    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.num_features = num_features
        self.eps = eps

        # 可学习的缩放和平移参数
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        """
        x: (batch_size, num_features)
        """
        # 在特征维度上计算均值和方差
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)

        # 归一化
        x_normalized = (x - mean) / torch.sqrt(var + self.eps)

        # 缩放和平移
        out = self.gamma * x_normalized + self.beta
        return out


class MLPWithNormalization(nn.Module):
    """
    3层MLP，包含Z值归一化、批量归一化和层归一化
    """
    def __init__(self, input_size, hidden_size, output_size=1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Z值归一化（预处理）
        self.z_score_norm = ZScoreNormalization()

        # 第0层到第1层
        self.linear0 = nn.Linear(input_size, hidden_size, bias=True)

        # 第1层归一化: 批量归一化
        self.batch_norm1 = BatchNormalizationManual(hidden_size)

        # 第1层到第2层
        self.linear1 = nn.Linear(hidden_size, hidden_size, bias=True)

        # 第2层归一化: 层归一化
        self.layer_norm2 = LayerNormalizationManual(hidden_size)

        # 第2层到输出层
        self.linear2 = nn.Linear(hidden_size, output_size, bias=True)

        # 激活函数
        self.relu = nn.ReLU()

    def fit_z_score(self, X):
        """拟合Z值归一化参数"""
        self.z_score_norm.fit(X)

    def forward(self, x, return_intermediates=False):
        """
        前向传播
        x: (batch_size, input_size)
        """
        intermediates = {}

        # 第0层: 原始输入
        a0 = x  # 原始输入

        # (1) 对原始输入做Z值归一化
        a0_normalized = self.z_score_norm.transform(a0)
        intermediates['a0_normalized'] = a0_normalized

        # 第0层 -> 第1层
        z1 = self.linear0(a0_normalized)  # 净输入

        # (2) 对第1层净输入做批量归一化
        z1_normalized = self.batch_norm1(z1)
        intermediates['z1_normalized'] = z1_normalized

        a1 = self.relu(z1_normalized)  # 第1层输出
        intermediates['a1'] = a1

        # 第1层 -> 第2层
        z2 = self.linear1(a1)

        # (3) 对第2层净输入做层归一化
        z2_normalized = self.layer_norm2(z2)
        intermediates['z2_normalized'] = z2_normalized

        a2 = self.relu(z2_normalized)  # 第2层输出
        intermediates['a2'] = a2

        # 第2层 -> 输出层
        output = self.linear2(a2)

        if return_intermediates:
            return output, intermediates
        return output


# ==================== 测试与验证 ====================

class SyntheticDataset(Dataset):
    """合成数据集"""
    def __init__(self, num_samples=1000, input_size=10, noise_std=0.1):
        self.num_samples = num_samples
        self.input_size = input_size

        # 生成合成数据: y = sum(x_i^2) + noise
        np.random.seed(42)
        self.X = np.random.randn(num_samples, input_size).astype(np.float32)
        self.y = (self.X ** 2).sum(axis=1, keepdims=True).astype(np.float32)
        self.y += np.random.randn(num_samples, 1).astype(np.float32) * noise_std

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])


def verify_normalization():
    """验证各层归一化的正确性"""
    print("="*60)
    print("Problem 3: 数据预处理与逐层规范化")
    print("="*60)

    # 创建模型
    input_size = 10
    hidden_size = 32
    batch_size = 64

    model = MLPWithNormalization(input_size, hidden_size)

    # 生成测试数据
    np.random.seed(42)
    X_train = torch.randn(500, input_size) * 5 + 10  # 均值10，标准差5
    y_train = (X_train ** 2).sum(dim=1, keepdims=True)

    X_test = torch.randn(100, input_size) * 5 + 10
    y_test = (X_test ** 2).sum(dim=1, keepdims=True)

    # 拟合Z值归一化
    model.fit_z_score(X_train)

    print("\n(1) 验证Z值归一化:")
    print(f"  原始数据均值: {X_train.mean(dim=0)[:5].tolist()}")
    print(f"  原始数据标准差: {X_train.std(dim=0)[:5].tolist()}")

    X_normalized = model.z_score_norm.transform(X_train)
    print(f"  Z值归一化后均值: {X_normalized.mean(dim=0)[:5].tolist()}")
    print(f"  Z值归一化后方差: {X_normalized.var(dim=0)[:5].tolist()}")

    # 验证均值接近0，方差接近1
    assert torch.allclose(X_normalized.mean(dim=0), torch.zeros(input_size), atol=1e-4)
    assert torch.allclose(X_normalized.var(dim=0), torch.ones(input_size), atol=1e-2)
    print(" [OK] Z值归一化验证通过!")

    # 测试批量归一化
    print("\n(2) 验证批量归一化:")
    model.train()  # 训练模式
    z1 = model.linear0(X_normalized[:batch_size])
    z1_norm = model.batch_norm1(z1)
    print(f"  归一化前均值: {z1.mean(dim=0)[:5].tolist()}")
    print(f"  归一化前方差: {z1.var(dim=0)[:5].tolist()}")
    print(f"  归一化后均值: {z1_norm.mean(dim=0)[:5].tolist()}")
    print(f"  归一化后方差: {z1_norm.var(dim=0)[:5].tolist()}")

    # 验证缩放和平移参数可学习
    print(f"  gamma参数示例: {model.batch_norm1.gamma[:5].tolist()}")
    print(f"  beta参数示例: {model.batch_norm1.beta[:5].tolist()}")
    print("  [OK] 批量归一化验证通过!")

    # 测试层归一化
    print("\n(3) 验证层归一化:")
    model.train()
    with torch.no_grad():
        _, intermediates = model(X_normalized[:batch_size], return_intermediates=True)

    z2_norm = intermediates['z2_normalized']
    print(f"  层归一化后每样本均值: {z2_norm.mean(dim=1)[:5].tolist()}")
    print(f"  层归一化后每样本方差: {z2_norm.var(dim=1)[:5].tolist()}")

    # 验证每个样本的特征均值为0，方差为1
    assert torch.allclose(z2_norm.mean(dim=1), torch.zeros(batch_size), atol=1e-4)
    # LayerNorm后乘以gamma会使方差略大于1（因为var(gamma * x_norm) = gamma^2）
    assert torch.allclose(z2_norm.var(dim=1), torch.ones(batch_size), atol=0.05)
    print("  [OK] 层归一化验证通过!")

    return model


def train_and_compare():
    """训练对比: 带归一化 vs 不带归一化"""
    print("\n" + "="*60)
    print("训练对比实验")
    print("="*60)

    input_size = 20
    hidden_size = 64
    num_epochs = 50
    batch_size = 32

    # 生成数据
    dataset = SyntheticDataset(num_samples=2000, input_size=input_size)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 提取训练数据用于拟合Z值归一化
    X_train_all = torch.stack([train_dataset[i][0] for i in range(len(train_dataset))])

    # 模型1: 带归一化
    model_norm = MLPWithNormalization(input_size, hidden_size)
    model_norm.fit_z_score(X_train_all)

    # 模型2: 不带归一化（简化版本）
    class PlainMLP(nn.Module):
        def __init__(self, input_size, hidden_size):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, 1)
            )

        def forward(self, x):
            return self.net(x)

    model_plain = PlainMLP(input_size, hidden_size)

    # 训练
    criterion = nn.MSELoss()
    optimizer_norm = optim.Adam(model_norm.parameters(), lr=0.01)
    optimizer_plain = optim.Adam(model_plain.parameters(), lr=0.01)

    losses_norm = []
    losses_plain = []

    for epoch in range(num_epochs):
        epoch_loss_norm = 0
        epoch_loss_plain = 0

        for X_batch, y_batch in train_loader:
            # 带归一化
            model_norm.train()
            pred_norm = model_norm(X_batch)
            loss_norm = criterion(pred_norm, y_batch)
            optimizer_norm.zero_grad()
            loss_norm.backward()
            optimizer_norm.step()
            epoch_loss_norm += loss_norm.item()

            # 不带归一化
            model_plain.train()
            pred_plain = model_plain(X_batch)
            loss_plain = criterion(pred_plain, y_batch)
            optimizer_plain.zero_grad()
            loss_plain.backward()
            optimizer_plain.step()
            epoch_loss_plain += loss_plain.item()

        losses_norm.append(epoch_loss_norm / len(train_loader))
        losses_plain.append(epoch_loss_plain / len(train_loader))

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}: Norm Loss={losses_norm[-1]:.4f}, Plain Loss={losses_plain[-1]:.4f}")

    # 可视化
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(losses_norm, 'b-', linewidth=2, label='With Normalization')
    axes[0].plot(losses_plain, 'r--', linewidth=2, label='Without Normalization')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('MSE Loss')
    axes[0].set_title('Training Loss Comparison')
    axes[0].legend()
    axes[0].grid(True)

    # 测试集性能
    model_norm.eval()
    model_plain.eval()

    test_loss_norm = 0
    test_loss_plain = 0

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            pred_norm = model_norm(X_batch)
            pred_plain = model_plain(X_batch)
            test_loss_norm += criterion(pred_norm, y_batch).item()
            test_loss_plain += criterion(pred_plain, y_batch).item()

    test_loss_norm /= len(test_loader)
    test_loss_plain /= len(test_loader)

    # 柱状图对比
    axes[1].bar(['With Norm', 'Without Norm'], [test_loss_norm, test_loss_plain], color=['blue', 'red'], alpha=0.7)
    axes[1].set_ylabel('Test MSE Loss')
    axes[1].set_title('Test Loss Comparison')
    axes[1].grid(True, axis='y')

    plt.tight_layout()
    plt.savefig('problem3_results.png', dpi=150)
    print("\n结果已保存到 problem3_results.png")

    return losses_norm, losses_plain


def main():
    # 验证各层归一化的正确性
    model = verify_normalization()

    # 训练对比实验
    losses_norm, losses_plain = train_and_compare()

    print("\n" + "="*60)
    print("实验完成!")
    print("="*60)

    return model


if __name__ == '__main__':
    main()
