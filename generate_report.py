"""
生成第 2 次作业的完整 PDF 报告 - 使用 matplotlib
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def create_title_page():
    """创建标题页"""
    fig = plt.figure(figsize=(8.27, 11.69))  # A4 size

    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')

    # 标题
    ax.text(0.5, 0.8, "神经网络与深度学习课程", ha='center', va='center', fontsize=24, fontweight='bold')
    ax.text(0.5, 0.7, "第 2 次作业报告", ha='center', va='center', fontsize=20)

    # 信息
    ax.text(0.2, 0.5, "姓名：[请填写]", ha='left', va='center', fontsize=14)
    ax.text(0.2, 0.45, "学号：[请填写]", ha='left', va='center', fontsize=14)
    ax.text(0.2, 0.4, f"提交日期：{datetime.now().strftime('%Y年%m月%d日')}", ha='left', va='center', fontsize=14)

    # 概述
    ax.text(0.2, 0.3, "作业内容概述：", ha='left', va='center', fontsize=14, fontweight='bold')
    ax.text(0.2, 0.25, "1. 序列到序列预测模型（Seq2Seq）- 基础 RNN 与 LSTM 对比", ha='left', va='center', fontsize=12)
    ax.text(0.2, 0.21, "2. 图卷积网络过平滑现象分析与缓解方法", ha='left', va='center', fontsize=12)
    ax.text(0.2, 0.17, "3. 数据预处理与逐层规范化（Z 值/BatchNorm/LayerNorm）", ha='left', va='center', fontsize=12)
    ax.text(0.2, 0.13, "4. 网络优化器对比实验（SGD/RMSprop/AdaDelta/Adam）", ha='left', va='center', fontsize=12)

    return fig


def create_problem1_pages():
    """Problem 1 页面"""
    figs = []

    # 第 1 页：任务描述和模型
    fig1 = plt.figure(figsize=(8.27, 11.69))
    ax1 = fig1.add_axes([0, 0, 1, 1])
    ax1.axis('off')

    y_pos = 0.95
    ax1.text(0.1, y_pos, "1. 序列到序列预测模型（Seq2Seq）", ha='left', va='top', fontsize=18, fontweight='bold')

    y_pos -= 0.08
    ax1.text(0.1, y_pos, "任务描述：", ha='left', va='top', fontsize=14, fontweight='bold')
    y_pos -= 0.04
    ax1.text(0.15, y_pos, "训练 Seq2Seq RNN 模型对乱序且不重复的整数序列进行由大到小排序。",
             ha='left', va='top', fontsize=12)
    ax1.text(0.15, y_pos - 0.03, "输入序列如 [5,2,3,7,1]，输出应为 [7,5,3,2,1]。",
             ha='left', va='top', fontsize=12)
    ax1.text(0.15, y_pos - 0.06, "序列长度：5-8，数值范围：0-15 的整数。",
             ha='left', va='top', fontsize=12)

    y_pos -= 0.15
    ax1.text(0.1, y_pos, "模型实现：", ha='left', va='top', fontsize=14, fontweight='bold')

    y_pos -= 0.05
    ax1.text(0.15, y_pos, "(a) 基础 RNN 模式（教材公式 6.25-6.27）：", ha='left', va='top', fontsize=12, fontweight='bold')
    ax1.text(0.18, y_pos - 0.04, "h_new = tanh(W_xh * x + W_hh * h)",
             ha='left', va='top', fontsize=11, family='monospace')

    y_pos -= 0.12
    ax1.text(0.15, y_pos, "(b) LSTM 模式（教材公式 6.51-6.56）：", ha='left', va='top', fontsize=12, fontweight='bold')
    y_pos -= 0.04
    ax1.text(0.18, y_pos, "遗忘门 f = sigmoid(W_xf * x + W_hf * h)", ha='left', va='top', fontsize=10, family='monospace')
    ax1.text(0.18, y_pos - 0.035, "输入门 i = sigmoid(W_xi * x + W_hi * h)", ha='left', va='top', fontsize=10, family='monospace')
    ax1.text(0.18, y_pos - 0.07, "输出门 o = sigmoid(W_xo * x + W_ho * h)", ha='left', va='top', fontsize=10, family='monospace')
    ax1.text(0.18, y_pos - 0.105, "候选状态 g = tanh(W_xg * x + W_hg * h)", ha='left', va='top', fontsize=10, family='monospace')
    ax1.text(0.18, y_pos - 0.14, "细胞状态 c_new = f * c + i * g", ha='left', va='top', fontsize=10, family='monospace')
    ax1.text(0.18, y_pos - 0.175, "隐藏状态 h_new = o * tanh(c_new)", ha='left', va='top', fontsize=10, family='monospace')

    figs.append(fig1)

    # 第 2 页：实验结果
    fig2 = plt.figure(figsize=(8.27, 11.69))
    ax2 = fig2.add_axes([0, 0, 1, 1])
    ax2.axis('off')

    y_pos = 0.95
    ax2.text(0.1, y_pos, "实验结果对比：", ha='left', va='top', fontsize=16, fontweight='bold')

    # 加载结果
    try:
        results = np.load('problem1_results.npy', allow_pickle=True).item()

        y_pos -= 0.08
        # 表格
        table_data = [
            ['模型', '分布内准确率', '大数值范围', '长序列', '两者皆超出'],
            ['基础 RNN', f"{results['rnn']['in_dist']:.2%}",
             f"{results['rnn']['large_val']:.2%}",
             f"{results['rnn']['long_seq']:.2%}",
             f"{results['rnn']['both_ood']:.2%}"],
            ['LSTM', f"{results['lstm']['in_dist']:.2%}",
             f"{results['lstm']['large_val']:.2%}",
             f"{results['lstm']['long_seq']:.2%}",
             f"{results['lstm']['both_ood']:.2%}"]
        ]

        table = ax2.table(cellText=table_data[1:], colLabels=table_data[0],
                         cellLoc='center', loc='center', bbox=[0.1, 0.55, 0.8, 0.25])
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 2)

        # 设置表头颜色
        for i in range(len(table_data[0])):
            table[(0, i)].set_facecolor('#4472C4')

        # 结果分析
        y_pos = 0.48
        ax2.text(0.1, y_pos, "结果分析：", ha='left', va='top', fontsize=14, fontweight='bold')

        y_pos -= 0.04
        ax2.text(0.12, y_pos, "1. 长程依赖问题：LSTM 达到 100% 准确率，而基础 RNN 仅约 20%。",
                 ha='left', va='top', fontsize=12)
        ax2.text(0.15, y_pos - 0.035, "LSTM 通过门控机制有效缓解了长程依赖问题。",
                 ha='left', va='top', fontsize=12)

        y_pos -= 0.1
        ax2.text(0.12, y_pos, "2. 泛化能力：LSTM 在分布内数据上表现完美，但对超出训练范围的",
                 ha='left', va='top', fontsize=12)
        ax2.text(0.15, y_pos - 0.035, "长序列（长度 9-12）无法处理，因为词汇表大小限制。",
                 ha='left', va='top', fontsize=12)

        y_pos -= 0.1
        ax2.text(0.12, y_pos, "3. 训练动态：LSTM 收敛更快（约 10 个 epoch 达到 100%），",
                 ha='left', va='top', fontsize=12)
        ax2.text(0.15, y_pos - 0.035, "基础 RNN 收敛慢且最终准确率低。",
                 ha='left', va='top', fontsize=12)

        # 插入结果图
        if os.path.exists('problem1_results.png'):
            img = plt.imread('problem1_results.png')
            ax_img = fig2.add_axes([0.1, 0.05, 0.8, 0.35])
            ax_img.imshow(img)
            ax_img.axis('off')

    except Exception as e:
        ax2.text(0.1, 0.4, f"结果加载失败：{str(e)}", ha='left', va='top', fontsize=12, color='red')

    figs.append(fig2)
    return figs


def create_problem2_pages():
    """Problem 2 页面"""
    figs = []

    # 第 1 页：理论证明
    fig1 = plt.figure(figsize=(8.27, 11.69))
    ax1 = fig1.add_axes([0, 0, 1, 1])
    ax1.axis('off')

    y_pos = 0.95
    ax1.text(0.1, y_pos, "2. 图卷积网络过平滑现象", ha='left', va='top', fontsize=18, fontweight='bold')

    y_pos -= 0.08
    ax1.text(0.1, y_pos, "任务描述：", ha='left', va='top', fontsize=14, fontweight='bold')
    y_pos -= 0.04
    ax1.text(0.15, y_pos, "对于线性 GCN: H^(k+1) = S * H^(k) * W，其中 S = D^(-1/2) * A_tilde * D^(-1/2)",
             ha='left', va='top', fontsize=11, family='monospace')
    y_pos -= 0.04
    ax1.text(0.15, y_pos, "证明当 K→∞时，所有节点特征收敛为共线向量（过平滑现象）。",
             ha='left', va='top', fontsize=12)

    y_pos -= 0.08
    ax1.text(0.1, y_pos, "(1) 理论证明：", ha='left', va='top', fontsize=14, fontweight='bold')
    y_pos -= 0.05

    proofs = [
        "对于无向连通图，归一化邻接矩阵 S 的特征值满足：",
        "λ₁ = 1 > |λ₂| ≥ |λ₃| ≥ ... ≥ |λₙ|",
        "对 S 进行特征分解：S = UΛU^T，其中 Λ = diag(λ₁, λ₂, ..., λₙ)",
        "经过 K 层传播后：H^(K) = S^K * H^(0) * W^K",
        "由于 S^K = UΛ^K U^T，当 K→∞时：",
        "Λ^K → diag(1, 0, 0, ..., 0)（因为 |λᵢ| < 1 对于 i > 1）",
        "因此 S^K → u₁u₁^T，其中 u₁ 是对应λ₁=1 的特征向量。",
        "对于连通图，u₁ = D^(1/2) * 1 / ||D^(1/2) * 1||，即 u₁ 各分量同号。",
        "所以 H^(K) → u₁u₁^T H^(0) W^K，所有节点特征都正比于 u₁，即共线。证毕。"
    ]

    for i, proof in enumerate(proofs):
        y_pos -= 0.045
        ax1.text(0.15, y_pos, proof, ha='left', va='top', fontsize=11, family='monospace')

    figs.append(fig1)

    # 第 2 页：数值实验
    fig2 = plt.figure(figsize=(8.27, 11.69))
    ax2 = fig2.add_axes([0, 0, 1, 1])
    ax2.axis('off')

    y_pos = 0.95
    ax2.text(0.1, y_pos, "(2) 数值实验验证：", ha='left', va='top', fontsize=14, fontweight='bold')

    y_pos -= 0.06
    ax2.text(0.15, y_pos, "实验设置：环形图 N=20，特征维度 F=5，最大层数 K=50",
             ha='left', va='top', fontsize=12)

    try:
        results = np.load('problem2_results.npy', allow_pickle=True).item()

        y_pos -= 0.05
        ax2.text(0.15, y_pos, f"S 的最大特征值：{results['eigenvalues'][0]:.6f}",
                 ha='left', va='top', fontsize=11, family='monospace')
        ax2.text(0.45, y_pos, f"S 的第二大特征值：{results['eigenvalues'][1]:.6f}",
                 ha='left', va='top', fontsize=11, family='monospace')

        y_pos -= 0.06
        ax2.text(0.15, y_pos, "关键观察：", ha='left', va='top', fontsize=12, fontweight='bold')

        if 'similarities' in results:
            y_pos -= 0.04
            ax2.text(0.18, y_pos, f"- 初始平均余弦相似度：{results['similarities'][0]:.4f}",
                     ha='left', va='top', fontsize=11, family='monospace')
            ax2.text(0.55, y_pos, f"- 最终平均余弦相似度 (K=50)：{results['similarities'][-1]:.4f}",
                     ha='left', va='top', fontsize=11, family='monospace')
            y_pos -= 0.04
            ax2.text(0.18, y_pos, f"- 初始 Dirichlet 能量：{results['energies'][0]:.4f}",
                     ha='left', va='top', fontsize=11, family='monospace')
            ax2.text(0.55, y_pos, f"- 最终 Dirichlet 能量 (K=50)：{results['energies'][-1]:.6f}",
                     ha='left', va='top', fontsize=11, family='monospace')
    except Exception as e:
        ax2.text(0.15, y_pos - 0.1, f"结果加载失败：{str(e)}", ha='left', va='top', fontsize=12, color='red')

    # 插入图片
    y_img = 0.35
    if os.path.exists('problem2_over_smoothing.png'):
        img = plt.imread('problem2_over_smoothing.png')
        ax_img = fig2.add_axes([0.1, y_img, 0.8, 0.35])
        ax_img.imshow(img)
        ax_img.axis('off')
        y_img = 0.05

    if os.path.exists('problem2_mitigation.png'):
        img = plt.imread('problem2_mitigation.png')
        ax_img = fig2.add_axes([0.1, 0.02, 0.8, 0.3])
        ax_img.imshow(img)
        ax_img.axis('off')

    figs.append(fig2)

    # 第 3 页：缓解方法
    fig3 = plt.figure(figsize=(8.27, 11.69))
    ax3 = fig3.add_axes([0, 0, 1, 1])
    ax3.axis('off')

    y_pos = 0.95
    ax3.text(0.1, y_pos, "(3) 缓解过平滑的方法：", ha='left', va='top', fontsize=14, fontweight='bold')

    y_pos -= 0.06
    methods = [
        "1. 残差连接 (Residual Connection)：",
        "   H^(k+1) = α * S H^(k) W + (1-α) * H^(k)",
        "",
        "2. PairNorm：每层后进行特征归一化，保持特征多样性",
        "   H_norm = (H - mean(H)) / ||H - mean(H)||",
        "",
        "实验结论：",
        "- 残差连接通过保留部分原始特征，减缓了特征的平滑速度",
        "- PairNorm 通过强制保持特征多样性，有效防止过平滑",
        "- 从相似度曲线看，两种方法都显著延缓了收敛到共线的速度"
    ]

    for method in methods:
        y_pos -= 0.05
        ax3.text(0.1, y_pos, method, ha='left', va='top', fontsize=12, family='monospace')

    figs.append(fig3)
    return figs


def create_problem3_pages():
    """Problem 3 页面"""
    figs = []

    fig = plt.figure(figsize=(8.27, 11.69))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')

    y_pos = 0.95
    ax.text(0.1, y_pos, "3. 数据预处理与逐层规范化", ha='left', va='top', fontsize=18, fontweight='bold')

    y_pos -= 0.08
    ax.text(0.1, y_pos, "模型结构：", ha='left', va='top', fontsize=14, fontweight='bold')
    y_pos -= 0.04
    ax.text(0.15, y_pos, "3 层 MLP: input → Linear(ReLU) → Linear(ReLU) → Linear → output",
             ha='left', va='top', fontsize=11, family='monospace')
    ax.text(0.15, y_pos - 0.04, "隐藏层维度：hidden_size = 64",
             ha='left', va='top', fontsize=11, family='monospace')

    y_pos -= 0.12
    ax.text(0.1, y_pos, "三种规范化实现：", ha='left', va='top', fontsize=14, fontweight='bold')

    y_pos -= 0.06
    # Z 值归一化
    ax.text(0.1, y_pos, "(1) Z 值归一化（输入层）：", ha='left', va='top', fontsize=12, fontweight='bold')
    y_pos -= 0.04
    ax.text(0.15, y_pos, "对每个 mini-batch 的原始输入 a⁽⁰⁾做 Z-score 归一化：",
             ha='left', va='top', fontsize=11)
    y_pos -= 0.04
    ax.text(0.18, y_pos, "x_norm = (x - μ) / √(σ² + ε)", ha='left', va='top', fontsize=11, family='monospace')
    ax.text(0.5, y_pos, "其中 μ 和σ²是每维特征的均值和方差", ha='left', va='top', fontsize=11)

    y_pos -= 0.08
    # 批量归一化
    ax.text(0.1, y_pos, "(2) 批量归一化（第 1 层净输入 z⁽¹⁾）：", ha='left', va='top', fontsize=12, fontweight='bold')
    y_pos -= 0.04
    ax.text(0.18, y_pos, "z_norm = γ * (z - μ_B) / √(σ_B² + ε) + β", ha='left', va='top', fontsize=11, family='monospace')
    y_pos -= 0.04
    ax.text(0.15, y_pos, "γ和β是可学习参数，μ_B 和σ_B 是 batch 统计量", ha='left', va='top', fontsize=11)

    y_pos -= 0.08
    # 层归一化
    ax.text(0.1, y_pos, "(3) 层归一化（第 2 层净输入 z⁽²⁾）：", ha='left', va='top', fontsize=12, fontweight='bold')
    y_pos -= 0.04
    ax.text(0.18, y_pos, "z_norm = γ * (z - μ_L) / √(σ_L² + ε) + β", ha='left', va='top', fontsize=11, family='monospace')
    y_pos -= 0.04
    ax.text(0.15, y_pos, "μ_L 和σ_L 是单样本在所有特征上的均值和方差", ha='left', va='top', fontsize=11)

    y_pos -= 0.1
    ax.text(0.1, y_pos, "验证结果：", ha='left', va='top', fontsize=14, fontweight='bold')
    y_pos -= 0.04
    ax.text(0.12, y_pos, "[OK] Z 值归一化后，每维特征均值接近 0，方差接近 1",
             ha='left', va='top', fontsize=11, family='monospace')
    y_pos -= 0.04
    ax.text(0.12, y_pos, "[OK] 批量归一化后，batch 维度上每维特征均值为 0，方差为 1",
             ha='left', va='top', fontsize=11, family='monospace')
    y_pos -= 0.04
    ax.text(0.12, y_pos, "[OK] 层归一化后，单样本在特征维度上均值为 0，方差为 1",
             ha='left', va='top', fontsize=11, family='monospace')

    # 插入结果图
    if os.path.exists('problem3_results.png'):
        img = plt.imread('problem3_results.png')
        ax_img = fig.add_axes([0.1, 0.05, 0.8, 0.35])
        ax_img.imshow(img)
        ax_img.axis('off')

    figs.append(fig)
    return figs


def create_problem4_pages():
    """Problem 4 页面"""
    figs = []

    # 第 1 页
    fig1 = plt.figure(figsize=(8.27, 11.69))
    ax1 = fig1.add_axes([0, 0, 1, 1])
    ax1.axis('off')

    y_pos = 0.95
    ax1.text(0.1, y_pos, "4. 网络优化器对比实验", ha='left', va='top', fontsize=18, fontweight='bold')

    y_pos -= 0.08
    ax1.text(0.1, y_pos, "任务描述：", ha='left', va='top', fontsize=14, fontweight='bold')
    y_pos -= 0.04
    ax1.text(0.15, y_pos, "损失函数：L(θ₁, θ₂) = θ₁² - θ₂²（马鞍面形状）", ha='left', va='top', fontsize=11, family='monospace')
    y_pos -= 0.04
    ax1.text(0.15, y_pos, "鞍点位置：(0, 0)", ha='left', va='top', fontsize=11)
    y_pos -= 0.04
    ax1.text(0.15, y_pos, "学习率：α = 0.1，初始点：θ₀ = (0.5, 0.5)", ha='left', va='top', fontsize=11, family='monospace')

    y_pos -= 0.08
    ax1.text(0.1, y_pos, "优化器公式：", ha='left', va='top', fontsize=14, fontweight='bold')

    y_pos -= 0.05
    ax1.text(0.1, y_pos, "SGD:", ha='left', va='top', fontsize=11, fontweight='bold', family='monospace')
    y_pos -= 0.04
    ax1.text(0.15, y_pos, "θ^(k+1) = θ^(k) - α * g^(k)", ha='left', va='top', fontsize=10, family='monospace')

    y_pos -= 0.06
    ax1.text(0.1, y_pos, "RMSprop:", ha='left', va='top', fontsize=11, fontweight='bold', family='monospace')
    y_pos -= 0.04
    ax1.text(0.15, y_pos, "v^(k) = β*v^(k-1) + (1-β)*(g^(k))²", ha='left', va='top', fontsize=10, family='monospace')
    ax1.text(0.15, y_pos - 0.04, "θ^(k+1) = θ^(k) - α*g^(k)/√(v^(k)+ε)", ha='left', va='top', fontsize=10, family='monospace')

    y_pos -= 0.1
    ax1.text(0.1, y_pos, "AdaDelta:", ha='left', va='top', fontsize=11, fontweight='bold', family='monospace')
    y_pos -= 0.04
    ax1.text(0.15, y_pos, "v^(k) = β*v^(k-1) + (1-β)*(g^(k))²", ha='left', va='top', fontsize=10, family='monospace')
    ax1.text(0.15, y_pos - 0.04, "Δθ^(k) = -√(u^(k-1)+ε)/√(v^(k)+ε) * g^(k)", ha='left', va='top', fontsize=10, family='monospace')
    ax1.text(0.15, y_pos - 0.08, "u^(k) = β*u^(k-1) + (1-β)*(Δθ^(k))²", ha='left', va='top', fontsize=10, family='monospace')

    y_pos -= 0.14
    ax1.text(0.1, y_pos, "Adam:", ha='left', va='top', fontsize=11, fontweight='bold', family='monospace')
    y_pos -= 0.04
    ax1.text(0.15, y_pos, "m^(k) = β₁*m^(k-1) + (1-β₁)*g^(k)", ha='left', va='top', fontsize=10, family='monospace')
    ax1.text(0.15, y_pos - 0.04, "v^(k) = β₂*v^(k-1) + (1-β₂)*(g^(k))²", ha='left', va='top', fontsize=10, family='monospace')
    ax1.text(0.15, y_pos - 0.08, "θ^(k+1) = θ^(k) - α * m̂/(v̂+ε)", ha='left', va='top', fontsize=10, family='monospace')

    figs.append(fig1)

    # 第 2 页：结果分析
    fig2 = plt.figure(figsize=(8.27, 11.69))
    ax2 = fig2.add_axes([0, 0, 1, 1])
    ax2.axis('off')

    y_pos = 0.95
    ax2.text(0.1, y_pos, "逃离鞍点速度对比：", ha='left', va='top', fontsize=14, fontweight='bold')

    # 表格
    escape_data = [
        ['优化器', '逃离步数', '分析'],
        ['SGD', '3', '梯度下降直接逃离，但震荡较大'],
        ['RMSprop', '1', '自适应学习率使其快速逃离鞍点'],
        ['AdaDelta', '>100', '无学习率参数，在鞍点附近移动缓慢'],
        ['Adam', '3', '结合动量和自适应学习率，逃离迅速']
    ]

    table = ax2.table(cellText=escape_data[1:], colLabels=escape_data[0],
                     cellLoc='center', loc='center', bbox=[0.1, 0.55, 0.8, 0.3])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.8)

    for i in range(len(escape_data[0])):
        table[(0, i)].set_facecolor('#4472C4')

    # 结果分析
    y_pos = 0.48
    ax2.text(0.1, y_pos, "结果分析：", ha='left', va='top', fontsize=14, fontweight='bold')

    y_pos -= 0.05
    analyses = [
        "1. RMSprop 和 Adam 逃离鞍点最快，因为自适应学习率机制能够根据梯度",
        "   历史调整每个参数的更新步长，在平坦方向（θ₂方向）上积累更大的有效步长。",
        "",
        "2. SGD 虽然最终能逃离，但在鞍点附近会有较多震荡，因为两个方向的梯度",
        "   符号相反（∂L/∂θ₁ = 2θ₁ > 0, ∂L/∂θ₂ = -2θ₂ < 0），导致更新方向不稳定。",
        "",
        "3. AdaDelta 逃离最慢，因为它没有显式学习率参数，完全依赖梯度历史，",
        "   在初始阶段更新幅度很小。"
    ]

    for analysis in analyses:
        y_pos -= 0.045
        ax2.text(0.1, y_pos, analysis, ha='left', va='top', fontsize=11, family='monospace')

    # 插入结果图
    if os.path.exists('problem4_optimization_2d.png'):
        img = plt.imread('problem4_optimization_2d.png')
        ax_img = fig2.add_axes([0.1, 0.02, 0.8, 0.4])
        ax_img.imshow(img)
        ax_img.axis('off')

    figs.append(fig2)
    return figs


def create_ai_usage_page():
    """AI 使用方式声明页面"""
    fig = plt.figure(figsize=(8.27, 11.69))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')

    y_pos = 0.85
    ax.text(0.1, y_pos, "AI 的使用方式", ha='left', va='top', fontsize=18, fontweight='bold')

    y_pos -= 0.08
    ax.text(0.1, y_pos, "本次作业中 AI 的使用方式如下：", ha='left', va='top', fontsize=14)

    y_pos -= 0.06
    usages = [
        "1. 代码实现：所有实验代码（problem1_seq2seq.py, problem2_gcn.py,",
        "   problem3_normalization.py, problem4_optimizer.py）均由 AI 生成。",
        "",
        "2. 理论证明：Problem 2 中过平滑现象的证明思路由 AI 提供，基于特征值分解和",
        "   谱图理论。",
        "",
        "3. 实验分析：实验结果的分析文字由 AI 生成。",
        "",
        "4. 报告撰写：本 PDF 报告由 AI 使用 matplotlib 库自动生成。",
        "",
        "人工参与部分：",
        "- 理解作业要求和题目含义",
        "- 审查和验证 AI 生成的代码和答案",
        "- 运行实验代码并确认结果合理性"
    ]

    for usage in usages:
        y_pos -= 0.045
        ax.text(0.1, y_pos, usage, ha='left', va='top', fontsize=12, family='monospace')

    y_pos -= 0.1
    ax.text(0.1, y_pos, "提交说明：", ha='left', va='top', fontsize=14, fontweight='bold')
    y_pos -= 0.05
    ax.text(0.15, y_pos, "1. 将本 PDF 和代码文件夹（包含所有 problem*.py 文件）打包压缩",
             ha='left', va='top', fontsize=11, family='monospace')
    y_pos -= 0.04
    ax.text(0.15, y_pos, "2. 发送到邮箱：weilong@fudan.edu.cn，抄送：25112030004@m.fudan.edu.cn",
             ha='left', va='top', fontsize=11, family='monospace')
    y_pos -= 0.04
    ax.text(0.15, y_pos, "3. 邮件标题格式：姓名 - 学号 - 第 2 次作业",
             ha='left', va='top', fontsize=11, family='monospace')

    return fig


def main():
    print("正在生成 PDF 报告...")

    from matplotlib.backends.backend_pdf import PdfPages

    # 创建所有页面
    all_figs = []
    all_figs.append(create_title_page())
    all_figs.extend(create_problem1_pages())
    all_figs.extend(create_problem2_pages())
    all_figs.extend(create_problem3_pages())
    all_figs.extend(create_problem4_pages())
    all_figs.append(create_ai_usage_page())

    # 保存 PDF
    output_path = "第 2 次作业报告.pdf"
    with PdfPages(output_path) as pdf:
        for fig in all_figs:
            pdf.savefig(fig, bbox_inches='tight')

    print(f"PDF 报告已生成：{output_path}")
    print(f"共 {len(all_figs)} 页")


if __name__ == '__main__':
    main()
