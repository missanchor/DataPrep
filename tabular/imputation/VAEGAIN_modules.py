import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tqdm import tqdm


# ==========================================
# 1. 神经网络组件
# ==========================================

class Encoder(nn.Module):
    def __init__(self, input_dim, h1, h2, latent_dim):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc_mean = nn.Linear(h2, latent_dim)
        self.fc_log_var = nn.Linear(h2, latent_dim)  # 输出 log(sigma^2)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        z_mean = self.fc_mean(h2)
        z_log_var = self.fc_log_var(h2)
        return z_mean, z_log_var


class Decoder(nn.Module):
    def __init__(self, latent_dim, h1, h2, output_dim):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc_mean = nn.Linear(h2, output_dim)
        self.fc_log_sigma = nn.Linear(h2, output_dim)  # 输出 log(sigma)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, z):
        h1 = F.relu(self.fc1(z))
        h2 = F.relu(self.fc2(h1))
        x_mean = self.fc_mean(h2)
        x_log_sigma = self.fc_log_sigma(h2)
        return x_mean, x_log_sigma


class Discriminator(nn.Module):
    def __init__(self, input_dim, h1, h2):
        super(Discriminator, self).__init__()
        # 输入是 Data + Hint，所以维度是 input_dim * 2
        self.fc1 = nn.Linear(input_dim * 2, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, input_dim)  # 输出每个维度的概率

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
        logits = self.fc3(h2)
        return torch.sigmoid(logits)


# ==========================================
# 2. 工具函数 (归一化与采样)
# ==========================================

def normalization(data):
    """计算归一化参数并转换数据到 [0, 1]"""
    _min = np.nanmin(data, axis=0)
    _max = np.nanmax(data, axis=0)
    _den = _max - _min
    _den[_den == 0] = 1e-6
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
    return np.random.uniform(0., 0.01, size=[batch_size, dim])


def sample_M(batch_size, dim, p):
    unif = np.random.uniform(0., 1., size=[batch_size, dim])
    return (unif > p).astype(float)


# ==========================================
# 3. 核心训练逻辑
# ==========================================

def gaussian_log_likelihood(x, mean, log_sigma):
    """计算高斯分布的对数似然"""
    log_sigma = torch.clamp(log_sigma, min=-10, max=5)

    sigma = torch.exp(log_sigma)
    dist = torch.distributions.Normal(mean, sigma)
    log_prob = dist.log_prob(x)
    return log_prob


def train_vaegain(encoder, decoder, discriminator, data_x, mask, params, device):
    """
    执行 VAE-GAIN 的训练循环 (带 Loss 显示)
    """
    no, dim = data_x.shape

    # 优化器定义
    opt_vae = optim.RMSprop(list(encoder.parameters()) + list(decoder.parameters()), lr=params['learning_rate'])
    opt_d = optim.Adam(discriminator.parameters())

    encoder.train()
    decoder.train()
    discriminator.train()

    batch_size = params['batch_size']

    pbar = tqdm(range(params['epoch']), desc="VAE-GAIN Training")

    for _ in pbar:
        # Shuffle index
        idx = np.random.permutation(no)

        # 用于记录当前 epoch 最后一个 batch 的 loss
        curr_d_loss = 0.0
        curr_g_loss = 0.0
        curr_vae_loss = 0.0

        for i in range(0, no, batch_size):
            if i + batch_size > no: break
            mb_idx = idx[i: i + batch_size]

            # 准备数据
            X_mb_np = data_x[mb_idx]
            M_mb_np = mask[mb_idx]

            # 采样 Noise, Mask, Hint
            Z_mb_np = sample_Z(batch_size, dim)  # 用于填补缺失值的初始噪音
            # Hint 向量生成
            H_mb_temp = sample_M(batch_size, dim, 1 - params['p_hint'])
            H_mb_np = M_mb_np * H_mb_temp

            # 构造输入：观测值保持，缺失值用随机噪音填充
            New_X_mb_np = M_mb_np * X_mb_np + (1 - M_mb_np) * Z_mb_np

            # 转 Tensor
            New_X = torch.tensor(New_X_mb_np, dtype=torch.float32).to(device)
            M = torch.tensor(M_mb_np, dtype=torch.float32).to(device)
            H = torch.tensor(H_mb_np, dtype=torch.float32).to(device)

            # ========================
            # 1. 训练 Discriminator
            # ========================
            for _ in range(params['discriminator_number']):
                opt_d.zero_grad()

                # VAE 前向传播生成填补值
                z_mean, z_log_var = encoder(New_X)
                # 重参数化技巧
                std = torch.exp(0.5 * z_log_var)
                eps = torch.randn_like(std)
                z = z_mean + eps * std

                x_hat_mean, _ = decoder(z)
                G_sample = x_hat_mean

                # 构造填补后的完整数据 (Generator output)
                Hat_New_X = New_X * M + G_sample * (1 - M)

                # Discriminator 判别
                D_prob = discriminator(Hat_New_X.detach(), H)  # detach 防止梯度传回 VAE

                # D Loss: Maximize log(D(real)) + log(1-D(fake))
                D_loss = -torch.mean(M * torch.log(D_prob + 1e-8) + (1 - M) * torch.log(1. - D_prob + 1e-8))

                D_loss.backward()
                torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
                opt_d.step()

                # 记录 D Loss
                curr_d_loss = D_loss.item()

            # ========================
            # 2. 训练 Generator (VAE)
            # ========================
            for _ in range(params['generator_number']):
                opt_vae.zero_grad()

                # VAE Forward
                z_mean, z_log_var = encoder(New_X)
                std = torch.exp(0.5 * z_log_var)
                eps = torch.randn_like(std)
                z = z_mean + eps * std

                x_hat_mean, x_hat_log_sigma = decoder(z)
                G_sample = x_hat_mean

                # 构造填补数据
                Hat_New_X = New_X * M + G_sample * (1 - M)

                # Discriminator 判别
                D_prob = discriminator(Hat_New_X, H)

                # --- Loss 计算 ---
                # 1. Adversarial Loss (Generator fooling Discriminator)
                G_loss_adv = -torch.mean((1 - M) * torch.log(D_prob + 1e-8))

                # 2. Reconstruction Loss (Log Likelihood)
                log_p_x_z = gaussian_log_likelihood(New_X, x_hat_mean, x_hat_log_sigma)
                reconstr_loss = -torch.sum(M * log_p_x_z, dim=1)  # Sum over dimensions

                # 3. KL Divergence
                kl_loss = -0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp(), dim=1)

                # 组合 VAE Loss
                vae_term = torch.mean(reconstr_loss + kl_loss) / (torch.mean(M) + 1e-8)
                Total_G_Loss = G_loss_adv + params['alpha'] * vae_term

                Total_G_Loss.backward()
                torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=1.0)
                opt_vae.step()

                # 记录 G Loss 和 VAE 部分的 Loss
                curr_g_loss = Total_G_Loss.item()
                curr_vae_loss = vae_term.item()

        # D_Loss: 判别器损失
        # G_Loss: 生成器总损失 (对抗 + VAE)
        pbar.set_postfix({
            'D_Loss': f"{curr_d_loss:.4f}",
            'G_Loss': f"{curr_g_loss:.4f}"
        })