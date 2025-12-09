import numpy as np
import torch
from dataprep.base import BaseEstimator
import dataprep.tabular.imputation.GAIN_module as gm

class GAINImputer(BaseEstimator):
    def __init__(self,
                 batch_size=128,
                 hint_rate=0.9,
                 alpha=100,
                 epoch=100,
                 device=None):
        self.batch_size = batch_size
        self.hint_rate = hint_rate
        self.alpha = alpha
        self.epoch = epoch
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')

        # 内部状态
        self.norm_parameters = None
        self.generator = None
        self.discriminator = None

    def train(self, data, missing_mask):
        """
        Args:
            data: np.array, 包含数据的矩阵 (可以包含 NaN，也可以是已经填0的，依据 mask)
            missing_mask: np.array, 0表示缺失, 1表示观测到 (与 data 形状相同)
        """
        data = np.array(data)
        missing_mask = np.array(missing_mask)
        no, dim = data.shape
        h_dim = int(dim)

        # 1. 数据归一化 (使用 module 中的函数)
        data_for_norm = data.copy()
        data_for_norm[missing_mask == 0] = np.nan
        norm_data, self.norm_parameters = gm.normalization(data_for_norm)

        # 将 NaN 填为 0 以输入网络
        norm_data_x = np.nan_to_num(norm_data, 0)

        # 2. 初始化网络
        self.generator = gm.GainGenerator(dim, h_dim).to(self.device)
        self.discriminator = gm.GainDiscriminator(dim, h_dim).to(self.device)

        # 3. 调用 Module 中的训练循环
        params = {
            'batch_size': self.batch_size,
            'epoch': self.epoch,
            'hint_rate': self.hint_rate,
            'alpha': self.alpha
        }

        print(f"Starting GAIN training on {self.device}...")
        gm.train_gain_model(
            self.generator,
            self.discriminator,
            norm_data_x,
            missing_mask,
            params,
            self.device
        )
        print("Training finished.")
        return self

    def predict(self, data):
        """
        Args:
            data: 原始数据
            missing_mask: 0表示缺失, 1表示观测到
        Returns:
            imputed_data: 填补后的完整数据
        """
        if self.generator is None:
            raise RuntimeError("Model needs to be trained first.")

        self.generator.eval()
        data = np.array(data)
        missing_mask = 1 - np.isnan(data)
        no, dim = data.shape

        # 1. 归一化
        # 使用训练时保存的参数
        norm_data = gm.normalization_with_parameter(data, self.norm_parameters)
        norm_data_x = np.nan_to_num(norm_data, 0)  # 缺失处填0

        # 2. 准备输入
        Z_mb = gm.uniform_sampler(0, 0.01, no, dim)
        M_mb = missing_mask
        X_mb = M_mb * norm_data_x + (1 - M_mb) * Z_mb

        X_mb_torch = torch.tensor(X_mb, dtype=torch.float32).to(self.device)
        M_mb_torch = torch.tensor(M_mb, dtype=torch.float32).to(self.device)

        # 3. 生成填补值
        with torch.no_grad():
            imputed_norm = self.generator(X_mb_torch, M_mb_torch).cpu().numpy()

        # 4. 组合观测值与生成值
        imputed_data_norm = M_mb * norm_data_x + (1 - M_mb) * imputed_norm

        # 5. 反归一化
        imputed_data = gm.renormalization(imputed_data_norm, self.norm_parameters)

        return imputed_data