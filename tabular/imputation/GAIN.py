import numpy as np
import torch
# 假设 GainGenerator, GainDiscriminator 和工具函数在 GAIN_modules.py 中
import dataprep.tabular.imputation.GAIN_modules as gm
from dataprep.tabular.imputation.base import BaseImputer  # 假设你有这个基类


class GAIN(BaseImputer):
    def __init__(self,
                 batch_size=128,
                 hint_rate=0.9,
                 alpha=100,
                 epoch=10000,
                 device=None):
        """
        GAIN Imputer Class
        Args:
            batch_size: 批大小
            hint_rate: 提示率 (Hint Rate)
            alpha: 用于平衡生成器损失中 MSE 部分的超参数
            epoch: 迭代次数
            device: 'cuda' 或 'cpu'
        """
        self.batch_size = batch_size
        self.hint_rate = hint_rate
        self.alpha = alpha
        self.epoch = epoch
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')

        # 内部状态
        self.norm_parameters = None
        self.generator = None
        self.discriminator = None

    def train(self, data: np.ndarray, missing_mask: np.ndarray = None) -> None:
        """
        训练 GAIN 模型。
        Args:
            data: np.array, 原始数据 (包含 np.nan)
            missing_mask: np.array, (可选) 0表示缺失, 1表示观测到。如果不传则自动根据 nan 生成。
        """
        # 创建临时目录保存模型 (如果基类有这个方法)
        if hasattr(self, '_create_temp_dir'):
            self._create_temp_dir(prefix="gain_train_")

        data = np.array(data)

        # 自动生成 mask
        if missing_mask is None:
            missing_mask = 1. - np.isnan(data)
        else:
            missing_mask = np.array(missing_mask)

        no, dim = data.shape
        h_dim = int(dim)  # 隐藏层维度通常设为输入维度

        # 1. 数据归一化 (使用 module 中的工具)
        data_for_norm = data.copy()
        # 确保计算 min/max 时忽略 NaN
        norm_data, self.norm_parameters = gm.normalization(data_for_norm)

        # 将 NaN 填充为 0 作为网络输入
        norm_data_x = np.nan_to_num(norm_data, 0)

        # 2. 初始化网络模型
        self.generator = gm.GainGenerator(dim, h_dim).to(self.device)
        self.discriminator = gm.GainDiscriminator(dim, h_dim).to(self.device)

        # 3. 封装训练参数
        params = {
            'batch_size': self.batch_size,
            'epoch': self.epoch,
            'hint_rate': self.hint_rate,
            'alpha': self.alpha
        }

        # 4. 调用 Module 中的核心训练算法
        gm.train_gain_algorithm(
            self.generator,
            self.discriminator,
            norm_data_x,
            missing_mask,
            params,
            self.device
        )

        # 保存模型检查点 (如果基类有这个方法)
        if hasattr(self, '_save_checkpoint'):
            self._save_checkpoint("gain_imputer_complete.pkl")

    def predict(self, data: np.ndarray) -> np.ndarray:
        """
        使用训练好的生成器进行数据填补。
        """
        if self.generator is None:
            raise RuntimeError("Model needs to be trained first. Call fit/train.")

        self.generator.eval()  # 切换到评估模式

        data = np.array(data)
        missing_mask = 1. - np.isnan(data)
        no, dim = data.shape

        # 1. 归一化输入数据
        norm_data = gm.normalization_with_parameter(data, self.norm_parameters)
        norm_data_x = np.nan_to_num(norm_data, 0)

        # 2. 准备 Tensor 输入
        # M_mb = missing_mask
        # Z_mb = Random Noise
        # X_mb = M * X + (1-M) * Z
        Z_mb = gm.sample_Z(no, dim)
        X_mb = missing_mask * norm_data_x + (1 - missing_mask) * Z_mb

        X_mb_torch = torch.tensor(X_mb, dtype=torch.float32).to(self.device)
        M_mb_torch = torch.tensor(missing_mask, dtype=torch.float32).to(self.device)

        # 3. 生成填补数据
        with torch.no_grad():
            imputed_norm_prob = self.generator(X_mb_torch, M_mb_torch).cpu().numpy()

        # 4. 组合观测数据与生成数据
        # GAIN 的逻辑：如果是观测值，保留原值；如果是缺失值，用生成值
        imputed_data_norm = missing_mask * norm_data_x + (1 - missing_mask) * imputed_norm_prob

        # 5. 反归一化
        imputed_data = gm.renormalization(imputed_data_norm, self.norm_parameters)

        return imputed_data