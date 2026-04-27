"""
Problem 2: 图卷积网络过平滑现象
(1) 证明过平滑现象
(2) 设计数值实验验证
(3) 提出缓解方法并验证
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
import networkx as nx


def compute_normalized_adjacency(A):
    """计算归一化邻接矩阵 S = D^(-1/2) * A_tilde * D^(-1/2)"""
    N = A.shape[0]
    # 添加自环
    A_tilde = A + np.eye(N)
    # 计算度矩阵
    D_tilde = np.diag(np.sum(A_tilde, axis=1))
    # 计算 D^(-1/2)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(D_tilde) + 1e-8))
    # 归一化邻接矩阵
    S = D_inv_sqrt @ A_tilde @ D_inv_sqrt
    return S


def gcn_propagation_linear(S, H0, W_list, K):
    """
    线性GCN的K层传播
    S: 归一化邻接矩阵 (N, N)
    H0: 初始特征 (N, F)
    W_list: 每层权重列表
    K: 层数
    """
    H = H0
    hidden_states = [H.copy()]

    for k in range(K):
        W = W_list[k] if k < len(W_list) else np.eye(H.shape[1])
        H = S @ H @ W
        hidden_states.append(H.copy())

    return H, hidden_states


def generate_random_graph(N, p=0.3, seed=None):
    """生成随机无向连通图"""
    if seed is not None:
        np.random.seed(seed)

    # 使用Erdos-Renyi随机图，确保连通
    while True:
        G = nx.erdos_renyi_graph(N, p)
        if nx.is_connected(G):
            break
        p = min(p + 0.1, 0.9)

    A = nx.to_numpy_array(G)
    return A, G


def generate_cycle_graph(N):
    """生成环形图（保证连通）"""
    G = nx.cycle_graph(N)
    A = nx.to_numpy_array(G)
    return A, G


def compute_feature_similarity(H):
    """计算节点特征之间的余弦相似度"""
    # 归一化特征
    H_norm = H / (np.linalg.norm(H, axis=1, keepdims=True) + 1e-8)
    # 余弦相似度矩阵
    sim = H_norm @ H_norm.T
    return sim


def compute_dirichlet_energy(H, S):
    """
    计算Dirichlet能量 (用于衡量过平滑)
    E(H) = sum_{i,j} A_ij ||h_i - h_j||^2
    """
    N = H.shape[0]
    energy = 0.0
    for i in range(N):
        for j in range(N):
            if S[i, j] > 0:  # 考虑邻接关系
                energy += np.linalg.norm(H[i] - H[j])**2
    return energy / 2  # 每条边计算两次


def compute_pairwise_distance_variance(H):
    """计算节点特征间距离的方差"""
    N = H.shape[0]
    dists = []
    for i in range(N):
        for j in range(i+1, N):
            dists.append(np.linalg.norm(H[i] - H[j]))
    return np.var(dists)


# ==================== 实验1: 验证过平滑现象 ====================

def experiment_over_smoothing():
    """
    实验：验证随着层数增加，节点特征趋于共线（过平滑）
    """
    print("\n" + "="*60)
    print("实验: 验证过平滑现象")
    print("="*60)

    # 参数设置
    N = 20  # 节点数
    F = 5   # 特征维度
    max_K = 50  # 最大层数

    # 生成图
    A, G = generate_cycle_graph(N)
    S = compute_normalized_adjacency(A)

    print(f"图结构: 环形图, N={N}, 边数={np.sum(A)//2}")

    # 检查S的特征值
    eigenvalues = np.linalg.eigvals(S)
    eigenvalues = np.sort(np.abs(eigenvalues))[::-1]
    print(f"S的最大特征值: {eigenvalues[0]:.6f}")
    print(f"S的第二大特征值: {eigenvalues[1]:.6f}")

    # 随机初始化特征
    np.random.seed(42)
    H0 = np.random.randn(N, F)

    # 每层使用相同权重（简化分析）
    W = np.eye(F)  # 单位矩阵，简化分析

    # 记录每层的度量
    similarities = []
    energies = []
    variances = []
    ranks = []

    H = H0.copy()
    for k in range(max_K):
        # 计算度量
        sim_matrix = compute_feature_similarity(H)
        # 平均相似度（排除对角线）
        avg_sim = (np.sum(sim_matrix) - N) / (N * (N - 1))
        similarities.append(avg_sim)

        energy = compute_dirichlet_energy(H, A)
        energies.append(energy)

        variance = compute_pairwise_distance_variance(H)
        variances.append(variance)

        # 计算特征矩阵的秩
        rank = np.linalg.matrix_rank(H, tol=1e-6)
        ranks.append(rank)

        # GCN传播
        H = S @ H @ W

    # 可视化结果
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. 平均余弦相似度
    axes[0, 0].plot(similarities, 'b-', linewidth=2)
    axes[0, 0].axhline(y=1.0, color='r', linestyle='--', label='Perfect Smoothing')
    axes[0, 0].set_xlabel('Layer K')
    axes[0, 0].set_ylabel('Average Cosine Similarity')
    axes[0, 0].set_title('Node Feature Similarity vs Layer Depth')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # 2. Dirichlet能量
    axes[0, 1].semilogy(energies, 'g-', linewidth=2)
    axes[0, 1].set_xlabel('Layer K')
    axes[0, 1].set_ylabel('Dirichlet Energy (log scale)')
    axes[0, 1].set_title('Dirichlet Energy vs Layer Depth')
    axes[0, 1].grid(True)

    # 3. 特征距离方差
    axes[1, 0].plot(variances, 'm-', linewidth=2)
    axes[1, 0].set_xlabel('Layer K')
    axes[1, 0].set_ylabel('Pairwise Distance Variance')
    axes[1, 0].set_title('Feature Diversity vs Layer Depth')
    axes[1, 0].grid(True)

    # 4. 特征矩阵秩
    axes[1, 1].plot(ranks, 'c-', linewidth=2)
    axes[1, 1].axhline(y=1, color='r', linestyle='--', label='Rank 1 (Collinear)')
    axes[1, 1].set_xlabel('Layer K')
    axes[1, 1].set_ylabel('Matrix Rank')
    axes[1, 1].set_title('Feature Matrix Rank vs Layer Depth')
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    plt.tight_layout()
    plt.savefig('problem2_over_smoothing.png', dpi=150)
    print("\n结果已保存到 problem2_over_smoothing.png")

    # 打印关键观察
    print(f"\n关键观察:")
    print(f"  初始平均相似度: {similarities[0]:.4f}")
    print(f"  最终平均相似度: {similarities[-1]:.4f}")
    print(f"  初始Dirichlet能量: {energies[0]:.4f}")
    print(f"  最终Dirichlet能量: {energies[-1]:.6f}")
    print(f"  初始特征秩: {ranks[0]}")
    print(f"  最终特征秩: {ranks[-1]}")

    return similarities, energies, variances, ranks


# ==================== 实验2: 特征值分析 ====================

def experiment_eigenvalue_analysis():
    """
    分析特征值衰减与过平滑的关系
    """
    print("\n" + "="*60)
    print("实验: 特征值分析")
    print("="*60)

    N = 20
    A, G = generate_cycle_graph(N)
    S = compute_normalized_adjacency(A)

    # 特征值分解
    eigenvalues, eigenvectors = np.linalg.eig(S)
    # 按特征值大小排序
    idx = np.argsort(np.abs(eigenvalues))[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    print("前10个特征值:")
    for i in range(min(10, N)):
        print(f"  λ_{i+1} = {np.abs(eigenvalues[i]):.6f}")

    # 可视化特征向量（作为初始特征）的传播
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    for idx, mode_idx in enumerate([0, 1, 2, 3, 4, 5]):
        ax = axes[idx // 3, idx % 3]

        # 以特征向量作为初始特征
        H0 = eigenvectors[:, mode_idx:mode_idx+1].real
        H = H0.copy()

        # 记录该模式的衰减
        norms = [np.linalg.norm(H)]
        for k in range(30):
            H = S @ H
            norms.append(np.linalg.norm(H))

        ax.semilogy(norms, linewidth=2)
        ax.set_xlabel('Layer K')
        ax.set_ylabel('Feature Norm (log scale)')
        ax.set_title(f'Mode {mode_idx+1}, λ={np.abs(eigenvalues[mode_idx]):.4f}')
        ax.grid(True)

    plt.tight_layout()
    plt.savefig('problem2_eigenvalue_decay.png', dpi=150)
    print("\n结果已保存到 problem2_eigenvalue_decay.png")

    return eigenvalues


# ==================== 实验3: 缓解过平滑的方法 ====================

def gcn_with_residual(S, H0, W_list, K, alpha=0.5):
    """
    带残差连接的GCN
    H^{k+1} = alpha * S @ H^k @ W + (1-alpha) * H^k
    """
    H = H0.copy()
    hidden_states = [H.copy()]

    for k in range(K):
        W = W_list[k] if k < len(W_list) else np.eye(H.shape[1])
        H_new = alpha * (S @ H @ W) + (1 - alpha) * H
        H = H_new
        hidden_states.append(H.copy())

    return H, hidden_states


def gcn_with_dropout(S, H0, W_list, K, dropout_rate=0.5, seed=42):
    """
    带Dropout的GCN（在传播过程中随机丢弃边）
    """
    np.random.seed(seed)
    H = H0.copy()
    hidden_states = [H.copy()]

    for k in range(K):
        W = W_list[k] if k < len(W_list) else np.eye(H.shape[1])

        # 随机丢弃边
        if k > 0:  # 第一层不dropout
            mask = (np.random.rand(*S.shape) > dropout_rate).astype(float)
            S_masked = S * mask
            # 重新归一化
            row_sums = S_masked.sum(axis=1, keepdims=True)
            S_masked = S_masked / (row_sums + 1e-8)
        else:
            S_masked = S

        H = S_masked @ H @ W
        hidden_states.append(H.copy())

    return H, hidden_states


def gcn_with_pairwise_normalization(S, H0, W_list, K):
    """
    PairNorm: 在每一层后进行特征归一化
    """
    H = H0.copy()
    hidden_states = [H.copy()]

    for k in range(K):
        W = W_list[k] if k < len(W_list) else np.eye(H.shape[1])
        H = S @ H @ W

        # PairNorm: 中心化并缩放
        H = H - H.mean(axis=0, keepdims=True)
        H = H / (np.linalg.norm(H, axis=1, keepdims=True) + 1e-8)

        hidden_states.append(H.copy())

    return H, hidden_states


def experiment_mitigation_methods():
    """
    对比不同缓解过平滑方法的效果
    """
    print("\n" + "="*60)
    print("实验: 缓解过平滑方法对比")
    print("="*60)

    N = 20
    F = 5
    max_K = 50

    A, G = generate_cycle_graph(N)
    S = compute_normalized_adjacency(A)

    np.random.seed(42)
    H0 = np.random.randn(N, F)
    W_list = [np.eye(F) for _ in range(max_K)]

    # 方法1: 基础GCN
    _, states_basic = gcn_propagation_linear(S, H0, W_list, max_K)
    sims_basic = [compute_feature_similarity(H).mean() for H in states_basic]

    # 方法2: 残差连接
    _, states_residual = gcn_with_residual(S, H0, W_list, max_K, alpha=0.5)
    sims_residual = [compute_feature_similarity(H).mean() for H in states_residual]

    # 方法3: PairNorm
    _, states_pairnorm = gcn_with_pairwise_normalization(S, H0, W_list, max_K)
    sims_pairnorm = [compute_feature_similarity(H).mean() for H in states_pairnorm]

    # 可视化对比
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 相似度对比
    axes[0].plot(sims_basic, 'b-', linewidth=2, label='Basic GCN')
    axes[0].plot(sims_residual, 'r--', linewidth=2, label='Residual Connection')
    axes[0].plot(sims_pairnorm, 'g-.', linewidth=2, label='PairNorm')
    axes[0].axhline(y=1.0, color='k', linestyle=':', alpha=0.5, label='Perfect Smoothing')
    axes[0].set_xlabel('Layer K')
    axes[0].set_ylabel('Average Cosine Similarity')
    axes[0].set_title('Over-smoothing Mitigation: Feature Similarity')
    axes[0].legend()
    axes[0].grid(True)

    # Dirichlet能量对比
    energies_basic = [compute_dirichlet_energy(H, A) for H in states_basic]
    energies_residual = [compute_dirichlet_energy(H, A) for H in states_residual]
    energies_pairnorm = [compute_dirichlet_energy(H, A) for H in states_pairnorm]

    axes[1].semilogy(energies_basic, 'b-', linewidth=2, label='Basic GCN')
    axes[1].semilogy(energies_residual, 'r--', linewidth=2, label='Residual Connection')
    axes[1].semilogy(energies_pairnorm, 'g-.', linewidth=2, label='PairNorm')
    axes[1].set_xlabel('Layer K')
    axes[1].set_ylabel('Dirichlet Energy (log scale)')
    axes[1].set_title('Over-smoothing Mitigation: Dirichlet Energy')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig('problem2_mitigation.png', dpi=150)
    print("\n结果已保存到 problem2_mitigation.png")

    # 打印对比结果
    print("\n最终层(K=50)的对比:")
    print(f"  基础GCN相似度: {sims_basic[-1]:.6f}")
    print(f"  残差连接相似度: {sims_residual[-1]:.6f}")
    print(f"  PairNorm相似度: {sims_pairnorm[-1]:.6f}")

    return {
        'basic': sims_basic,
        'residual': sims_residual,
        'pairnorm': sims_pairnorm
    }


def main():
    """运行所有实验"""
    # 实验1: 验证过平滑现象
    similarities, energies, variances, ranks = experiment_over_smoothing()

    # 实验2: 特征值分析
    eigenvalues = experiment_eigenvalue_analysis()

    # 实验3: 缓解方法对比
    mitigation_results = experiment_mitigation_methods()

    # 保存结果
    results = {
        'similarities': similarities,
        'energies': energies,
        'variances': variances,
        'ranks': ranks,
        'eigenvalues': eigenvalues,
        'mitigation': mitigation_results
    }
    np.save('problem2_results.npy', results, allow_pickle=True)
    print("\n所有结果已保存到 problem2_results.npy")

    return results


if __name__ == '__main__':
    main()
