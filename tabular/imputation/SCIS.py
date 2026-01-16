import numpy as np
import torch
import dataprep.tabular.imputation.SCIS_modules as sm
from dataprep.tabular.imputation.base import BaseImputer

class SCIS(BaseImputer):
    def __init__(self,
                 batch_size=128,
                 hint_rate=0.9,
                 alpha=100,
                 value=2,
                 epoch=100,
                 epsilon=1.4,
                 thre_value=0.001,
                 initial_value=20000,
                 guarantee=0.95,
                 device=None):
        self.batch_size = batch_size
        self.hint_rate = hint_rate
        self.alpha = alpha
        self.value = value
        self.epoch = epoch
        self.epsilon = epsilon
        self.thre_value = thre_value
        self.initial_value = initial_value
        self.guarantee = guarantee
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')

        # 内部状态
        self.norm_parameters = None
        self.generator = None
        self.discriminator = None

    def train(self, data: np.ndarray, missing_mask: np.ndarray) -> None:
        """
        训练 SCIS 模型。
        Args:
            data: np.array, 原始数据
            missing_mask: np.array, 0表示缺失, 1表示观测到
        """
        self._create_temp_dir(prefix="scis_train_")
        data = np.array(data)
        missing_mask = np.array(missing_mask)
        no, dim = data.shape
        h_dim = int(dim)

        # 1. 归一化 (调用 module)
        data_for_norm = data.copy()
        data_for_norm[missing_mask == 0] = np.nan
        norm_data, self.norm_parameters = sm.normalization(data_for_norm)
        norm_data_x = np.nan_to_num(norm_data, 0)

        # 2. 初始化网络
        self.generator = sm.GainGenerator(dim, h_dim).to(self.device)
        self.discriminator = sm.GainDiscriminator(dim, h_dim).to(self.device)

        # 3. 调用 Module 中的核心算法
        params = {
            'batch_size': self.batch_size,
            'epoch': self.epoch,
            'hint_rate': self.hint_rate,
            'alpha': self.alpha,
            'value': self.value,
            'epsilon': self.epsilon,
            'thre_value': self.thre_value,
            'initial_value': self.initial_value,
            'guarantee': self.guarantee
        }

        print(f"Starting SCIS training on {self.device}...")
        sm.train_scis_algorithm(
            self.generator,
            self.discriminator,
            norm_data_x,
            missing_mask,
            params,
            self.device
        )
        self._save_checkpoint("scis_imputer_complete.pkl")

        print("Training finished and parameters saved to temp dir.")

    def predict(self, data: np.ndarray) -> np.ndarray:
        """
        预测/填补数据。
        """
        if self.generator is None:
            raise RuntimeError("Model needs to be trained first.")

        self.generator.eval()
        data = np.array(data)
        missing_mask = 1 - np.isnan(data)
        no, dim = data.shape

        # 1. 归一化
        norm_data = sm.normalization_with_parameter(data, self.norm_parameters)
        norm_data_x = np.nan_to_num(norm_data, 0)

        # 2. 准备输入
        Z_mb = sm.uniform_sampler(0, 0.01, no, dim)
        M_mb = missing_mask
        X_mb = M_mb * norm_data_x + (1 - M_mb) * Z_mb

        X_mb_torch = torch.tensor(X_mb, dtype=torch.float32).to(self.device)
        M_mb_torch = torch.tensor(M_mb, dtype=torch.float32).to(self.device)

        # 3. 生成
        with torch.no_grad():
            imputed_norm = self.generator(X_mb_torch, M_mb_torch).cpu().numpy()

        # 4. 组合与反归一化
        imputed_data_norm = M_mb * norm_data_x + (1 - M_mb) * imputed_norm
        imputed_data = sm.renormalization(imputed_data_norm, self.norm_parameters)

        return imputed_data