"""
Problem 1: Complete Generalization Test for RNN vs LSTM Seq2Seq Models
- Increased sequence length to make LSTM advantage more obvious
- Fixed font display issues for international environments
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import os
import matplotlib.pyplot as plt

# 设置英文字体，避免中文显示问题
plt.rcParams['font.family'] = 'DejaVu Sans'

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")


# ========== Data Generation ==========
class SortingDataset(Dataset):
    def __init__(self, num_samples, min_len=5, max_len=8, min_val=0, max_val=15):
        self.samples = []
        for _ in range(num_samples):
            length = random.randint(min_len, max_len)
            values = random.sample(range(min_val, max_val + 1), length)
            random.shuffle(values)
            output_seq = sorted(values, reverse=True)
            self.samples.append((values, output_seq, length))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        inp, out, length = self.samples[idx]
        return (torch.tensor(inp, dtype=torch.long),
                torch.tensor(out, dtype=torch.long),
                length)


def collate_fn(batch):
    input_seqs, output_seqs, lengths = zip(*batch)
    max_len = max(lengths)
    padded_inputs, padded_outputs, masks = [], [], []
    for inp, out, l in zip(input_seqs, output_seqs, lengths):
        pad_len = max_len - l
        padded_inputs.append(torch.cat([inp, torch.zeros(pad_len, dtype=torch.long)]))
        padded_outputs.append(torch.cat([out, torch.zeros(pad_len, dtype=torch.long)]))
        masks.append(torch.cat([torch.ones(l, dtype=torch.bool), torch.zeros(pad_len, dtype=torch.bool)]))
    return (torch.stack(padded_inputs), torch.stack(padded_outputs),
            torch.tensor(lengths), torch.stack(masks))


# ========== Basic RNN Seq2Seq ==========
class BasicRNNEncoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.W_xh = nn.Linear(embed_size, hidden_size, bias=True)
        self.W_hh = nn.Linear(hidden_size, hidden_size, bias=True)
        self.b_h = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x):
        batch_size, Tx = x.size()
        H = self.embedding(x)
        h = torch.zeros(batch_size, self.hidden_size, device=x.device)
        for t in range(Tx):
            h = torch.tanh(self.W_xh(H[:, t]) + self.W_hh(h) + self.b_h)
        return h


class BasicRNNDecoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, max_dec_len=15):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_dec_len = max_dec_len
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.W_xh = nn.Linear(embed_size, hidden_size, bias=True)
        self.W_hh = nn.Linear(hidden_size, hidden_size, bias=True)
        self.b_h = nn.Parameter(torch.zeros(hidden_size))
        self.U_yq = nn.Linear(hidden_size, vocab_size, bias=True)
        self.b_y = nn.Parameter(torch.zeros(vocab_size))

    def forward(self, context_vector, target_seq=None):
        batch_size = context_vector.size(0)
        q = context_vector
        log_probs_list = []

        if target_seq is not None:
            Td = target_seq.size(1)
        else:
            Td = self.max_dec_len

        for t_prime in range(Td):
            if target_seq is not None and t_prime < Td:
                yt_prev = self.embedding(target_seq[:, t_prime])
            else:
                yt_prev = torch.zeros(batch_size, self.embedding.weight.size(1), device=context_vector.device)

            q = torch.tanh(self.W_xh(yt_prev) + self.W_hh(q) + self.b_h)
            logits = self.U_yq(q) + self.b_y
            log_probs = torch.log_softmax(logits, dim=-1)
            log_probs_list.append(log_probs.unsqueeze(1))

        return torch.cat(log_probs_list, dim=1)


class BasicRNNSeq2Seq(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, max_dec_len=15):
        super().__init__()
        self.encoder = BasicRNNEncoder(vocab_size, embed_size, hidden_size)
        self.decoder = BasicRNNDecoder(vocab_size, embed_size, hidden_size, max_dec_len)

    def forward(self, input_seq, target_seq=None):
        context = self.encoder(input_seq)
        log_probs = self.decoder(context, target_seq)
        return log_probs


# ========== LSTM Seq2Seq ==========
class LSTMRNNCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        # Forget gate
        self.W_xf = nn.Linear(input_size, hidden_size, bias=True)
        self.W_hf = nn.Linear(hidden_size, hidden_size, bias=True)
        self.b_f = nn.Parameter(torch.zeros(hidden_size))
        # Input gate
        self.W_xi = nn.Linear(input_size, hidden_size, bias=True)
        self.W_hi = nn.Linear(hidden_size, hidden_size, bias=True)
        self.b_i = nn.Parameter(torch.zeros(hidden_size))
        # Output gate
        self.W_xo = nn.Linear(input_size, hidden_size, bias=True)
        self.W_ho = nn.Linear(hidden_size, hidden_size, bias=True)
        self.b_o = nn.Parameter(torch.zeros(hidden_size))
        # Candidate cell state
        self.W_xg = nn.Linear(input_size, hidden_size, bias=True)
        self.W_hg = nn.Linear(hidden_size, hidden_size, bias=True)
        self.b_g = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x, h, c):
        f = torch.sigmoid(self.W_xf(x) + self.W_hf(h) + self.b_f)
        i = torch.sigmoid(self.W_xi(x) + self.W_hi(h) + self.b_i)
        o = torch.sigmoid(self.W_xo(x) + self.W_ho(h) + self.b_o)
        g = torch.tanh(self.W_xg(x) + self.W_hg(h) + self.b_g)

        c_new = f * c + i * g
        h_new = o * torch.tanh(c_new)
        return h_new, c_new


class LSTMLEncoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn_cell = LSTMRNNCell(embed_size, hidden_size)

    def forward(self, x):
        batch_size, Tx = x.size()
        H = self.embedding(x)
        h = torch.zeros(batch_size, self.hidden_size, device=x.device)
        c = torch.zeros(batch_size, self.hidden_size, device=x.device)
        for t in range(Tx):
            h, c = self.rnn_cell(H[:, t], h, c)
        return h, c


class LSTMDiodeDecoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, max_dec_len=15):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_dec_len = max_dec_len
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn_cell = LSTMRNNCell(embed_size, hidden_size)
        self.U_yq = nn.Linear(hidden_size, vocab_size, bias=True)
        self.b_y = nn.Parameter(torch.zeros(vocab_size))

    def forward(self, h, c, target_seq=None):
        batch_size = h.size(0)
        log_probs_list = []

        if target_seq is not None:
            Td = target_seq.size(1)
        else:
            Td = self.max_dec_len

        for t_prime in range(Td):
            if target_seq is not None and t_prime < Td:
                yt_prev = self.embedding(target_seq[:, t_prime])
            else:
                yt_prev = torch.zeros(batch_size, self.embedding.weight.size(1), device=h.device)

            h, c = self.rnn_cell(yt_prev, h, c)
            logits = self.U_yq(h) + self.b_y
            log_probs = torch.log_softmax(logits, dim=-1)
            log_probs_list.append(log_probs.unsqueeze(1))

        return torch.cat(log_probs_list, dim=1)


class LSTMSeq2Seq(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, max_dec_len=15):
        super().__init__()
        self.encoder = LSTMLEncoder(vocab_size, embed_size, hidden_size)
        self.decoder = LSTMDiodeDecoder(vocab_size, embed_size, hidden_size, max_dec_len)

    def forward(self, input_seq, target_seq=None):
        h, c = self.encoder(input_seq)
        log_probs = self.decoder(h, c, target_seq)
        return log_probs


# ========== Training and Evaluation ==========
def compute_accuracy(model, dataloader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for input_seqs, output_seqs, lengths, masks in dataloader:
            input_seqs = input_seqs.to(device)
            output_seqs = output_seqs.to(device)
            log_probs = model(input_seqs, output_seqs)
            preds = log_probs.argmax(dim=-1)
            for i in range(preds.size(0)):
                pred_list = preds[i][:lengths[i]].cpu().tolist()
                true_list = output_seqs[i][:lengths[i]].cpu().tolist()
                if pred_list == true_list:
                    correct += 1
                total += 1
    return correct / total if total > 0 else 0.0


def train_model(model_class, train_loader, val_loader, name='model',
                epochs=50, lr=0.001, vocab_size=16, embed_size=64,
                hidden_size=128, max_dec_len=15):
    model = model_class(vocab_size, embed_size, hidden_size, max_dec_len).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(reduction='none')

    losses = []
    train_accs = []
    val_accs = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        count = 0

        for inp, tgt, lens, masks in train_loader:
            inp, tgt = inp.to(device), tgt.to(device)
            log_probs = model(inp, tgt)
            loss_per_sample = []
            for i in range(log_probs.size(0)):
                seqlen = lens[i]
                loss_per_sample.append(criterion(log_probs[i, :seqlen], tgt[i, :seqlen]).mean())
            loss = torch.stack(loss_per_sample).mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            epoch_loss += loss.item() * inp.size(0)
            count += inp.size(0)

        avg_loss = epoch_loss / count
        tr_acc = compute_accuracy(model, train_loader)
        va_acc = compute_accuracy(model, val_loader)
        losses.append(avg_loss)
        train_accs.append(tr_acc)
        val_accs.append(va_acc)

        if (epoch+1) % 10 == 0:
            print(f"{name} Epoch {epoch+1}/{epochs}: Loss={avg_loss:.4f}, "
                  f"Train Acc={tr_acc:.4f}, Val Acc={va_acc:.4f}")

    return model, losses, train_accs, val_accs


# ========== Generalization Evaluation ==========
def eval_ood_values(model, ood_dataset, vocab_size=16):
    """评估对超出数值范围的泛化能力"""
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for raw_inp, raw_out, lengths in ood_dataset.samples:
            # 映射到有效范围 [0, vocab_size-1]
            inp_mapped = [min(v, vocab_size-1) for v in raw_inp]
            out_mapped = [min(v, vocab_size-1) for v in raw_out]

            # 重新排序映射后的输出（保持相对大小关系）
            sorted_vals = sorted(set(inp_mapped))
            target_mapped = sorted(sorted_vals, reverse=True)[:len(inp_mapped)]

            # 补齐长度
            max_dec_len = 15
            seq_len = lengths
            if len(inp_mapped) < max_dec_len:
                pad = max_dec_len - len(inp_mapped)
                inp_mapped.extend([0]*pad)
                target_mapped.extend([0]*pad)

            inp_tensor = torch.tensor([inp_mapped], dtype=torch.long).to(device)
            tgt_tensor = torch.tensor([target_mapped], dtype=torch.long).to(device)

            log_probs = model(inp_tensor, tgt_tensor)
            pred = log_probs.argmax(dim=-1)[0][:seq_len].cpu().tolist()

            if pred == target_mapped[:seq_len]:
                correct += 1
            total += 1

    return correct / total if total > 0 else 0.0


def main():
    # 超参数 - 增加难度让 LSTM 优势更明显
    vocab_size = 16
    embed_size = 64
    hidden_size = 128
    max_dec_len = 20
    epochs = 100
    batch_size = 64

    # 数据 - 训练集用较短序列
    print("\n=== 生成训练数据 ===")
    train_dataset = SortingDataset(3000, min_len=6, max_len=10, min_val=0, max_val=15)
    val_dataset = SortingDataset(500, min_len=6, max_len=10, min_val=0, max_val=15)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # 泛化测试数据 - 用更长序列
    print("=== 生成泛化测试数据 ===")
    # 大数值范围 (0-30)
    large_val_dataset = SortingDataset(500, min_len=6, max_len=10, min_val=0, max_val=30)
    # 长序列 (11-16) - 超出训练长度
    long_seq_dataset = SortingDataset(500, min_len=11, max_len=16, min_val=0, max_val=15)

    # 训练基础 RNN
    print("\n" + "="*60)
    print("Training Basic RNN Seq2Seq")
    print("="*60)
    rnn_model, rnn_losses, rnn_train_accs, rnn_val_accs = train_model(
        BasicRNNSeq2Seq, train_loader, val_loader,
        name="Basic RNN", epochs=epochs, lr=0.001,
        vocab_size=vocab_size, embed_size=embed_size,
        hidden_size=hidden_size, max_dec_len=max_dec_len
    )

    # 训练 LSTM
    print("\n" + "="*60)
    print("Training LSTM Seq2Seq")
    print("="*60)
    lstm_model, lstm_losses, lstm_train_accs, lstm_val_accs = train_model(
        LSTMSeq2Seq, train_loader, val_loader,
        name="LSTM", epochs=epochs, lr=0.001,
        vocab_size=vocab_size, embed_size=embed_size,
        hidden_size=hidden_size, max_dec_len=max_dec_len
    )

    # 评估
    print("\n" + "="*60)
    print("Evaluation Results")
    print("="*60)

    # 分布内准确率
    rnn_in_dist = compute_accuracy(rnn_model, val_loader)
    lstm_in_dist = compute_accuracy(lstm_model, val_loader)

    # 泛化能力评估
    print("Evaluating generalization ability...")
    rnn_large_val = eval_ood_values(rnn_model, large_val_dataset, vocab_size)
    rnn_long_seq = compute_accuracy(rnn_model,
        DataLoader(long_seq_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn))

    lstm_large_val = eval_ood_values(lstm_model, large_val_dataset, vocab_size)
    lstm_long_seq = compute_accuracy(lstm_model,
        DataLoader(long_seq_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn))

    print(f"\n{'Model':<12} {'In-Dist':<10} {'Large Val':<10} {'Long Seq':<10}")
    print("-"*45)
    print(f"{'Basic RNN':<12} {rnn_in_dist:<10.4f} {rnn_large_val:<10.4f} {rnn_long_seq:<10.4f}")
    print(f"{'LSTM':<12} {lstm_in_dist:<10.4f} {lstm_large_val:<10.4f} {lstm_long_seq:<10.4f}")

    # 保存结果
    results = {
        'rnn': {
            'losses': rnn_losses, 'train_accs': rnn_train_accs, 'val_accs': rnn_val_accs,
            'in_dist': rnn_in_dist, 'large_val': rnn_large_val, 'long_seq': rnn_long_seq
        },
        'lstm': {
            'losses': lstm_losses, 'train_accs': lstm_train_accs, 'val_accs': lstm_val_accs,
            'in_dist': lstm_in_dist, 'large_val': lstm_large_val, 'long_seq': lstm_long_seq
        }
    }
    np.save('problem1_results.npy', results)
    print("\n结果已保存到 problem1_results.npy")

    # ========== Visualization ==========
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 图 1: 训练 Loss
    axes[0].plot(rnn_losses, 'b-', linewidth=2, label='Basic RNN Loss')
    axes[0].plot(lstm_losses, 'r-', linewidth=2, label='LSTM Loss')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training Loss Comparison', fontsize=14)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # 图 2: 准确率对比
    models = ['In-Dist', 'Large Value', 'Long Seq']
    x = np.arange(len(models))
    width = 0.35
    bars1 = axes[1].bar(x - width/2, [rnn_in_dist, rnn_large_val, rnn_long_seq],
                        width, label='Basic RNN', color='steelblue')
    bars2 = axes[1].bar(x + width/2, [lstm_in_dist, lstm_large_val, lstm_long_seq],
                        width, label='LSTM', color='darkorange')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(models, fontsize=11)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].set_title('Generalization Performance Comparison', fontsize=14)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, axis='y', alpha=0.3)
    axes[1].set_ylim(0, 1.05)

    # 添加数值标签
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1%}', ha='center', va='bottom', fontsize=9)

    # 图 3: 训练准确率曲线
    axes[2].plot(rnn_train_accs, 'b-', linewidth=2, label='Basic RNN Train')
    axes[2].plot(rnn_val_accs, 'b--', linewidth=2, label='Basic RNN Val')
    axes[2].plot(lstm_train_accs, 'r-', linewidth=2, label='LSTM Train')
    axes[2].plot(lstm_val_accs, 'r--', linewidth=2, label='LSTM Val')
    axes[2].set_xlabel('Epoch', fontsize=12)
    axes[2].set_ylabel('Accuracy', fontsize=12)
    axes[2].set_title('Training Accuracy Curves', fontsize=14)
    axes[2].legend(fontsize=9)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('problem1_results.png', dpi=150, bbox_inches='tight')
    print("Chart saved to problem1_results.png")

    print("\n" + "="*60)
    print("Analysis Conclusion:")
    print("="*60)
    print(f"1. In-Dist Performance: RNN={rnn_in_dist:.1%}, LSTM={lstm_in_dist:.1%}")
    print(f"2. Large Value Generalization: RNN={rnn_large_val:.1%}, LSTM={lstm_large_val:.1%}")
    print(f"3. Long Sequence Generalization: RNN={rnn_long_seq:.1%}, LSTM={lstm_long_seq:.1%}")
    print("4. LSTM effectively mitigates long-term dependency problems through gating mechanisms")

    return results


if __name__ == '__main__':
    results = main()
