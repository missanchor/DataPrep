import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tqdm import tqdm


# ==========================================
# 1. 神经网络组件 (Neural Networks)
# ==========================================

class GainGenerator(nn.Module):
    def __init__(self, dim, h_dim):
        super(GainGenerator, self).__init__()
        self.fc1 = nn.Linear(dim * 2, h_dim)  # Input: Data + Mask
        self.fc2 = nn.Linear(h_dim, h_dim)
        self.fc3 = nn.Linear(h_dim, dim)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, m):
        # Concatenate Data and Mask
        inputs = torch.cat([x, m], dim=1)
        h1 = F.relu(self.fc1(inputs))
        h2 = F.relu(self.fc2(h1))
        # MinMax normalized output [0, 1]
        return torch.sigmoid(self.fc3(h2))


class GainDiscriminator(nn.Module):
    def __init__(self, dim, h_dim):
        super(GainDiscriminator, self).__init__()
        self.fc1 = nn.Linear(dim * 2, h_dim)  # Input: Data + Hint
        self.fc2 = nn.Linear(h_dim, h_dim)
        self.fc3 = nn.Linear(h_dim, dim)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, h):
        # Concatenate Data and Hint
        inputs = torch.cat([x, h], dim=1)
        h1 = F.relu(self.fc1(inputs))
        h2 = F.relu(self.fc2(h1))
        return torch.sigmoid(self.fc3(h2))  # Probability output


# ==========================================
# 2. 工具函数 (Utils)
# ==========================================

def normalization(data):
    """Min-Max 归一化，处理 NaN"""
    _min = np.nanmin(data, axis=0)
    _max = np.nanmax(data, axis=0)
    _den = _max - _min
    _den[_den == 0] = 1e-6  # 防止除以0
    norm_data = (data - _min) / _den
    norm_parameters = {'min': _min, 'max': _max, 'den': _den}
    return norm_data, norm_parameters


def normalization_with_parameter(data, norm_parameters):
    """使用已有参数进行归一化"""
    return (data - norm_parameters['min']) / norm_parameters['den']


def renormalization(norm_data, norm_parameters):
    """反归一化"""
    return norm_data * norm_parameters['den'] + norm_parameters['min']


def sample_Z(batch_size, dim):
    """生成随机噪声 Z"""
    return np.random.uniform(0., 0.01, size=[batch_size, dim])


def sample_M(batch_size, dim, p):
    """生成 Hint 向量所需的随机掩码"""
    unif_random_matrix = np.random.uniform(0., 1., size=[batch_size, dim])
    binary_random_matrix = 1. * (unif_random_matrix > p)
    return binary_random_matrix


# ==========================================
# 3. 核心训练流程 (Core Algorithms)
# ==========================================

def train_gain_algorithm(generator, discriminator, data_x, mask, params, device):
    """
    执行 GAIN 的标准训练循环 (带 Loss 显示)
    Args:
        data_x: 归一化后的数据 (已将NaN填为0)
        mask: 掩码矩阵 (1=Observed, 0=Missing)
    """
    no, dim = data_x.shape

    # 优化器
    opt_g = optim.Adam(generator.parameters())
    opt_d = optim.Adam(discriminator.parameters())

    print(f"Starting GAIN training on {device}...")

    pbar = tqdm(range(params['epoch']), desc="GAIN Training")

    for it in pbar:
        # 1. Mini-batch generation
        idx = np.random.permutation(no)
        batch_idx = idx[:params['batch_size']]

        X_mb = data_x[batch_idx, :]
        M_mb = mask[batch_idx, :]

        # Sample random vectors
        Z_mb = sample_Z(params['batch_size'], dim)

        # Sample hint vectors
        H_mb_temp = sample_M(params['batch_size'], dim, 1 - params['hint_rate'])
        H_mb = M_mb * H_mb_temp

        # Combine random vectors with observed vectors
        X_mb_with_noise = M_mb * X_mb + (1 - M_mb) * Z_mb

        # Convert to Torch Tensors
        X_mb_torch = torch.tensor(X_mb_with_noise, dtype=torch.float32).to(device)
        M_mb_torch = torch.tensor(M_mb, dtype=torch.float32).to(device)
        H_mb_torch = torch.tensor(H_mb, dtype=torch.float32).to(device)
        X_original_torch = torch.tensor(X_mb, dtype=torch.float32).to(device)

        # -----------------------------------
        # Train Discriminator
        # -----------------------------------
        opt_d.zero_grad()

        G_sample = generator(X_mb_torch, M_mb_torch)
        Hat_X = X_mb_torch * M_mb_torch + G_sample * (1 - M_mb_torch)
        D_prob = discriminator(Hat_X.detach(), H_mb_torch)

        D_loss = -torch.mean(M_mb_torch * torch.log(D_prob + 1e-8) + \
                             (1 - M_mb_torch) * torch.log(1. - D_prob + 1e-8))

        D_loss.backward()
        opt_d.step()

        # -----------------------------------
        # Train Generator
        # -----------------------------------
        opt_g.zero_grad()

        G_sample = generator(X_mb_torch, M_mb_torch)
        Hat_X = X_mb_torch * M_mb_torch + G_sample * (1 - M_mb_torch)
        D_prob = discriminator(Hat_X, H_mb_torch)

        G_loss_temp = -torch.mean((1 - M_mb_torch) * torch.log(D_prob + 1e-8))
        MSE_loss = torch.mean((M_mb_torch * X_original_torch - M_mb_torch * G_sample) ** 2) / torch.mean(M_mb_torch)

        G_loss = G_loss_temp + params['alpha'] * MSE_loss

        G_loss.backward()
        opt_g.step()

        # .item() 用于从 tensor 中取出数值，:.4f 表示保留4位小数
        pbar.set_postfix({
            'G_Loss': f"{G_loss.item():.4f}",
            'D_Loss': f"{D_loss.item():.4f}",
            'MSE': f"{MSE_loss.item():.4f}"
        })

    print("Training Complete.")