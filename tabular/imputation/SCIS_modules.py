import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tqdm import tqdm


# ==========================================
# 1. 神经网络组件
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
        return torch.sigmoid(self.fc3(h2))

    def forward_with_params(self, x, m, params):
        """支持传入外部参数（用于 Hessian 扰动计算）"""
        w1, b1, w2, b2, w3, b3 = params
        inputs = torch.cat([x, m], dim=1)
        h1 = F.relu(F.linear(inputs, w1, b1))
        h2 = F.relu(F.linear(h1, w2, b2))
        return torch.sigmoid(F.linear(h2, w3, b3))


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
        return self.fc3(h2)


# ==========================================
# 2. 工具函数
# ==========================================

def normalization(data):
    """Min-Max 归一化"""
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
    return norm_data * norm_parameters['den'] + norm_parameters['min']


def uniform_sampler(low, high, rows, cols):
    return np.random.uniform(low, high, size=[rows, cols])


def sinkhorn_loss_torch(x, y, epsilon=1.4, niter=20, p=2):
    """Sinkhorn 距离损失"""
    batch_size = x.size(0)
    x_col = x.unsqueeze(1)
    y_lin = y.unsqueeze(0)
    c = torch.sum(torch.abs(x_col - y_lin) ** p, dim=2)
    mu = torch.ones(batch_size, device=x.device) / batch_size
    nu = torch.ones(batch_size, device=x.device) / batch_size

    def M(u, v): return (-c + u.unsqueeze(1) + v.unsqueeze(0)) / epsilon

    u, v = torch.zeros_like(mu), torch.zeros_like(nu)
    for _ in range(niter):
        u = epsilon * (torch.log(mu) - torch.logsumexp(M(u, v), dim=1)) + u
        v = epsilon * (torch.log(nu) - torch.logsumexp(M(u, v).t(), dim=1)) + v
    return torch.sum(torch.exp(M(u, v)) * c)


def compute_hessian_diag(loss, params, device):
    """计算 Hessian 近似对角元素"""
    grads = torch.autograd.grad(loss, params, create_graph=True)
    diags = []
    for g in grads:
        g_flat = g.view(-1, 1)
        hess_approx = torch.matmul(g_flat, g_flat.t())
        eye = torch.eye(hess_approx.size(0), device=device)
        hess_inv = torch.linalg.inv(hess_approx + eye)
        diags.append(torch.diagonal(hess_inv).view(g.shape))
    return diags


# ==========================================
# 3. 核心训练流程
# ==========================================

def _train_step_gain(generator, discriminator, x_mb, m_mb, opt_g, opt_d, params, device):
    """单步训练 (已修改：返回 Loss 值)"""
    # 1. 准备输入
    z_mb = torch.rand_like(x_mb) * 0.01
    h_mb = m_mb * (torch.rand_like(m_mb) < params['hint_rate']).float()
    x_in = m_mb * x_mb + (1 - m_mb) * z_mb

    # 2. 训练判别器 (Discriminator)
    opt_d.zero_grad()
    g_sample = generator(x_in, m_mb)
    hat_x = m_mb * x_mb + (1 - m_mb) * g_sample
    d_prob = torch.sigmoid(discriminator(hat_x.detach(), h_mb))

    d_loss = -torch.mean(m_mb * torch.log(d_prob + 1e-8) + (1 - m_mb) * torch.log(1. - d_prob + 1e-8))
    d_loss.backward()
    opt_d.step()

    # 3. 训练生成器 (Generator)
    opt_g.zero_grad()
    g_sample = generator(x_in, m_mb)
    hat_x = m_mb * x_mb + (1 - m_mb) * g_sample
    d_prob = torch.sigmoid(discriminator(hat_x, h_mb))

    # 计算各项 Loss
    loss_g_temp = -torch.mean((1 - m_mb) * torch.log(d_prob + 1e-8))
    mse_loss = torch.mean((m_mb * x_mb - m_mb * g_sample) ** 2) / (torch.mean(m_mb) + 1e-8)

    # 总 G Loss (包含 Sinkhorn Loss)
    g_loss = loss_g_temp + params['alpha'] * mse_loss
    if 'value' in params and params['value'] > 0:
        # 假设 sinkhorn_loss_torch 在当前作用域可用
        g_loss += params['value'] * sinkhorn_loss_torch(x_mb, g_sample, epsilon=params['epsilon'])

    g_loss.backward()
    opt_g.step()

    # 返回 float 类型的 loss 值
    return g_loss.item(), d_loss.item(), mse_loss.item()


def train_scis_algorithm(generator, discriminator, data_x, mask, params, device):
    """执行 SCIS 三阶段训练：初始 -> 搜索 -> 重训练"""
    no, dim = data_x.shape
    opt_g, opt_d = optim.Adam(generator.parameters()), optim.Adam(discriminator.parameters())

#1.初始训练
    initial_num = min(params['initial_value'], int(no * 0.5)) if params['initial_value'] >= no else params[
        'initial_value']
    perm = np.random.permutation(no)
    init_idx, val_idx = perm[:initial_num], perm[initial_num: min(no, initial_num * 2)]

    print(f"Phase 1: Initial training ({initial_num} samples)...")

    pbar = tqdm(range(params['epoch']), desc="Initial Phase")

    for _ in pbar:
        idx = np.random.permutation(initial_num)
        # 记录每个 epoch 最后一个 batch 的 loss 用于显示
        current_g_loss, current_d_loss, current_mse = 0, 0, 0

        for i in range(0, initial_num, params['batch_size']):
            if i + params['batch_size'] > initial_num: break
            mb_idx = init_idx[idx[i:i + params['batch_size']]]

            # 【修改点】：接收返回值
            g_loss, d_loss, mse_loss = _train_step_gain(
                generator, discriminator,
                torch.tensor(data_x[mb_idx], dtype=torch.float32).to(device),
                torch.tensor(mask[mb_idx], dtype=torch.float32).to(device),
                opt_g, opt_d, params, device
            )
            current_g_loss, current_d_loss, current_mse = g_loss, d_loss, mse_loss

        # 【修改点】：更新进度条后缀
        pbar.set_postfix({
            'G': f"{current_g_loss:.4f}",
            'D': f"{current_d_loss:.4f}",
            'MSE': f"{current_mse:.4f}"
        })

    # 2. 样本量搜索 (Hessian)
    print("Phase 2: SCIS Search...")
    x_val = torch.tensor(data_x[val_idx[:128]], dtype=torch.float32).to(device)
    m_val = torch.tensor(mask[val_idx[:128]], dtype=torch.float32).to(device)
    hessian_loss = torch.mean((m_val * x_val - m_val * generator(
        m_val * x_val + (1 - m_val) * torch.rand_like(x_val) * 0.01, m_val)) ** 2) / (torch.mean(m_val) + 1e-9)

    try:
        h_diags = compute_hessian_diag(hessian_loss, list(generator.parameters()), device)
        up_num, down_num = no, initial_num
        median_num = int((up_num + down_num) / 2)
        epsilon_val = (np.exp(5 / params['epsilon']) / np.power(params['epsilon'], np.floor(dim / 2))) ** 2

        while median_num != down_num and median_num != up_num:
            predict_within = 0
            var_N = [d * (1. / median_num - 1. / no) for d in h_diags]
            var_n = [d * (1. / initial_num - 1. / median_num) for d in h_diags]

            for _ in range(20):
                p_N, p_n = [], []
                for i, p in enumerate(list(generator.parameters())):
                    p_N.append(torch.normal(p, torch.sqrt(torch.abs(var_N[i]) * epsilon_val + 1e-10)))
                    p_n.append(torch.normal(p, torch.sqrt(torch.abs(var_n[i]) * epsilon_val + 1e-10)))
                with torch.no_grad():
                    # 模拟注入
                    x_in = m_val * x_val + (1 - m_val) * (torch.rand_like(x_val) * 0.01)
                    rmse_N = torch.sqrt(
                        torch.mean((m_val * x_val - m_val * generator.forward_with_params(x_in, m_val, p_N)) ** 2))
                    rmse_n = torch.sqrt(
                        torch.mean((m_val * x_val - m_val * generator.forward_with_params(x_in, m_val, p_n)) ** 2))
                    if abs(rmse_n - rmse_N) < params['thre_value']: predict_within += 1
                    print(f"Diff: {abs(rmse_n - rmse_N).item():.4f}, Threshold: {params['thre_value']}")

            if predict_within > params['guarantee'] * 20:
                up_num = median_num
            else:
                down_num = median_num
            median_num = int((up_num + down_num) / 2)
        estimated_n = median_num
        print(f"Estimated N: {estimated_n}")
    except RuntimeError:
        print("Hessian failed, using full data.")
        estimated_n = no

    # 3. 重训练
    final_n = min(estimated_n, no)
    print(f"\nPhase 3: Retraining ({final_n} samples)...")
    train_idx = np.random.permutation(no)[:final_n]

    pbar_retrain = tqdm(range(params['epoch']), desc="Retraining Phase")

    for _ in pbar_retrain:
        idx = np.random.permutation(final_n)
        current_g_loss, current_d_loss, current_mse = 0, 0, 0

        for i in range(0, final_n, params['batch_size']):
            if i + params['batch_size'] > final_n: break
            mb_idx = train_idx[idx[i:i + params['batch_size']]]

            g_loss, d_loss, mse_loss = _train_step_gain(
                generator, discriminator,
                torch.tensor(data_x[mb_idx], dtype=torch.float32).to(device),
                torch.tensor(mask[mb_idx], dtype=torch.float32).to(device),
                opt_g, opt_d, params, device
            )
            current_g_loss, current_d_loss, current_mse = g_loss, d_loss, mse_loss

        pbar_retrain.set_postfix({
            'G': f"{current_g_loss:.4f}",
            'D': f"{current_d_loss:.4f}",
            'MSE': f"{current_mse:.4f}"
        })