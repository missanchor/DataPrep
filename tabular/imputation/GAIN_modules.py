import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tqdm import tqdm


# ==========================================
# 1. 神经网络定义 (Neural Networks)
# ==========================================

class GainGenerator(nn.Module):
    def __init__(self, dim, h_dim):
        super(GainGenerator, self).__init__()
        self.fc1 = nn.Linear(dim * 2, h_dim)
        self.fc2 = nn.Linear(h_dim, h_dim)
        self.fc3 = nn.Linear(h_dim, dim)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, m):
        inputs = torch.cat([x, m], dim=1)
        h1 = F.relu(self.fc1(inputs))
        h2 = F.relu(self.fc2(h1))
        return torch.sigmoid(self.fc3(h2))  # MinMax normalized output


class GainDiscriminator(nn.Module):
    def __init__(self, dim, h_dim):
        super(GainDiscriminator, self).__init__()
        self.fc1 = nn.Linear(dim * 2, h_dim)
        self.fc2 = nn.Linear(h_dim, h_dim)
        self.fc3 = nn.Linear(h_dim, dim)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, h):
        inputs = torch.cat([x, h], dim=1)
        h1 = F.relu(self.fc1(inputs))
        h2 = F.relu(self.fc2(h1))
        return self.fc3(h2)  # Logits


# ==========================================
# 2. 工具函数 (Utils & Loss)
# ==========================================

def normalization(data):
    """Normalize data to [0, 1] range."""
    _min = np.nanmin(data, axis=0)
    _max = np.nanmax(data, axis=0)
    _den = _max - _min
    _den[_den == 0] = 1e-6
    norm_data = (data - _min) / _den
    norm_parameters = {'min': _min, 'max': _max, 'den': _den}
    return norm_data, norm_parameters

def normalization_with_parameter(data, norm_parameters):
    return (data - norm_parameters['min']) / norm_parameters['den']

def renormalization(norm_data, norm_parameters):
    """Renormalize data from [0, 1] back to original range."""
    min_val = norm_parameters['min']
    max_val = norm_parameters['max']
    den = max_val - min_val
    renorm_data = norm_data * den + min_val
    return renorm_data


def sample_batch_index(total, batch_size):
    return np.random.permutation(total)[:batch_size]


def binary_sampler(p, rows, cols):
    unif_random_matrix = np.random.uniform(0., 1., size=[rows, cols])
    return 1 * (unif_random_matrix < p)


def uniform_sampler(low, high, rows, cols):
    return np.random.uniform(low, high, size=[rows, cols])


def sinkhorn_loss_torch(x, y, m, epsilon=1.4, niter=20, p=2):
    """PyTorch implementation of Sinkhorn Loss"""
    # 简化版 Sinkhorn，计算两组分布的差异
    batch_size = x.size(0)

    # Cost matrix
    x_col = x.unsqueeze(1)
    y_lin = y.unsqueeze(0)
    # Mask handling for cost (simplified for brevity)
    c = torch.sum(torch.abs(x_col - y_lin) ** p, dim=2)

    mu = torch.ones(batch_size, device=x.device) / batch_size
    nu = torch.ones(batch_size, device=x.device) / batch_size

    def M(u, v):
        return (-c + u.unsqueeze(1) + v.unsqueeze(0)) / epsilon

    u = torch.zeros_like(mu)
    v = torch.zeros_like(nu)

    for _ in range(niter):
        u = epsilon * (torch.log(mu) - torch.logsumexp(M(u, v), dim=1)) + u
        v = epsilon * (torch.log(nu) - torch.logsumexp(M(u, v).t(), dim=1)) + v

    pi = torch.exp(M(u, v))
    cost = torch.sum(pi * c)
    return cost


# ==========================================
# 3. 训练循环逻辑 (Training Loop)
# ==========================================

def train_gain_model(generator, discriminator, data_x, data_m, params, device):
    """
    执行 GAIN 的完整训练循环。
    data_x: 归一化后的数据 (np.array)
    data_m: 缺失掩码 (np.array)
    params: 字典，包含 batch_size, epoch, alpha, hint_rate 等
    """
    optimizer_G = optim.Adam(generator.parameters())
    optimizer_D = optim.Adam(discriminator.parameters())

    no, dim = data_x.shape
    batch_size = params['batch_size']

    generator.train()
    discriminator.train()

    # 转换为 Tensor

    for _ in tqdm(range(params['epoch']), desc="GAIN Training"):
        # Shuffle indices
        idx_list = np.random.permutation(no)

        for i in range(0, no, batch_size):
            if i + batch_size > no: break
            mb_idx = idx_list[i: i + batch_size]

            # 1. 准备 Batch 数据
            X_mb_np = data_x[mb_idx, :]
            M_mb_np = data_m[mb_idx, :]

            # 采样随机噪音 Z 和 Hint 向量 H
            Z_mb_np = uniform_sampler(0, 0.01, batch_size, dim)
            H_mb_temp = binary_sampler(params['hint_rate'], batch_size, dim)
            H_mb_np = M_mb_np * H_mb_temp

            # 输入到 Generator 的数据：观测值 + 噪音填充
            X_mb_input_np = M_mb_np * X_mb_np + (1 - M_mb_np) * Z_mb_np

            # 转为 Tensor
            X_mb = torch.tensor(X_mb_input_np, dtype=torch.float32).to(device)
            M_mb = torch.tensor(M_mb_np, dtype=torch.float32).to(device)
            H_mb = torch.tensor(H_mb_np, dtype=torch.float32).to(device)
            X_orig = torch.tensor(X_mb_np, dtype=torch.float32).to(device)  # 原始归一化数据(缺失处补0)

            # ---------------------
            # Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()

            G_sample = generator(X_mb, M_mb)
            # Hat_X: 观测部分保持原样，缺失部分用生成的填充
            Hat_X = M_mb * X_orig + (1 - M_mb) * G_sample

            D_prob = torch.sigmoid(discriminator(Hat_X.detach(), H_mb))

            # D Loss
            D_loss = -torch.mean(M_mb * torch.log(D_prob + 1e-8) + (1 - M_mb) * torch.log(1. - D_prob + 1e-8))

            D_loss.backward()
            optimizer_D.step()

            # ---------------------
            # Train Generator
            # ---------------------
            optimizer_G.zero_grad()

            G_sample = generator(X_mb, M_mb)
            Hat_X = M_mb * X_orig + (1 - M_mb) * G_sample
            D_prob = torch.sigmoid(discriminator(Hat_X, H_mb))

            # G Loss 1
            G_loss_temp = -torch.mean((1 - M_mb) * torch.log(D_prob + 1e-8))

            # G Loss 2
            MSE_loss = torch.mean((M_mb * X_mb - M_mb * G_sample) ** 2) / (torch.mean(M_mb) + 1e-8)

            # 总 Loss
            G_loss = G_loss_temp + params['alpha'] * MSE_loss

            G_loss.backward()
            optimizer_G.step()