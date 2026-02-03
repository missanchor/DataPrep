import numpy as np
import torch
from dataprep.tabular.imputation.base import BaseImputer
import dataprep.tabular.imputation.VAEGAIN_modules as vm

class VAEGAIN(BaseImputer):
    def __init__(self,
                 batch_size=8,
                 hint_rate=0.9,
                 alpha=10,
                 epoch=100,
                 learning_rate=0.002,
                 latent_size=20,
                 device=None):
        self.batch_size = batch_size
        self.hint_rate = hint_rate
        self.alpha = alpha
        self.epoch = epoch
        self.learning_rate = learning_rate
        self.latent_size = latent_size
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')

        # 内部状态
        self.norm_parameters = None
        self.encoder = None
        self.decoder = None
        self.discriminator = None

    def train(self, data: np.ndarray, missing_mask: np.ndarray) -> None:
        """
        Args:
            data: np.array, 原始数据 (包含 NaN)
        """
        self._create_temp_dir(prefix="vaegain_train_")
        data = np.array(data)
        no, dim = data.shape

        # 1. 数据归一化
        norm_data, self.norm_parameters = vm.normalization(data)
        norm_data_x = np.nan_to_num(norm_data, 0)

        # 2. 网络结构参数
        # 对应原代码的 hidden layer 设置
        enc_h1, enc_h2 = 50, 20
        dec_h1, dec_h2 = 50, 20

        # 3. 初始化网络
        self.encoder = vm.Encoder(dim, enc_h1, enc_h2, self.latent_size).to(self.device)
        self.decoder = vm.Decoder(self.latent_size, dec_h1, dec_h2, dim).to(self.device)
        self.discriminator = vm.Discriminator(dim, enc_h1, enc_h2).to(self.device)

        # 4. 准备参数并训练
        params = {
            'batch_size': self.batch_size,
            'epoch': self.epoch,
            'p_hint': self.hint_rate,
            'alpha': self.alpha,
            'learning_rate': self.learning_rate,
            'discriminator_number': 1,
            'generator_number': 1
        }

        print(f"Starting VAE-GAIN training on {self.device}...")
        vm.train_vaegain(
            self.encoder,
            self.decoder,
            self.discriminator,
            norm_data_x,
            missing_mask,
            params,
            self.device
        )
        self._save_checkpoint("vaegain_imputer_complete.pkl")

        print("Training finished and parameters saved to temp dir.")

    def predict(self, data: np.ndarray) -> np.ndarray:
        """
        Args:
            data: 原始数据
        Returns:
            imputed_data: 填补后的完整数据
        """
        if self.encoder is None or self.decoder is None:
            raise RuntimeError("Model needs to be trained first.")

        self.encoder.eval()
        self.decoder.eval()

        data = np.array(data)
        missing_mask = 1 - np.isnan(data)
        no, dim = data.shape

        # 1. 归一化
        norm_data = vm.normalization_with_parameter(data, self.norm_parameters)
        norm_data_x = np.nan_to_num(norm_data, 0)

        # 2. 准备输入
        Z_mb = vm.sample_Z(no, dim)
        M_mb = missing_mask
        X_mb = norm_data_x
        # 构造混合输入用于 Encoder
        New_X = M_mb * X_mb + (1 - M_mb) * Z_mb

        New_X_torch = torch.tensor(New_X, dtype=torch.float32).to(self.device)

        # 3. 生成填补值
        with torch.no_grad():
            z_mean, z_log_var = self.encoder(New_X_torch)
            std = torch.exp(0.5 * z_log_var)
            eps = torch.randn_like(std)
            z = z_mean + eps * std

            x_hat_mean, _ = self.decoder(z)
            imputed_norm = x_hat_mean.cpu().numpy()

        # 4. 组合观测值与生成值
        imputed_data_norm = M_mb * norm_data_x + (1 - M_mb) * imputed_norm

        # 5. 反归一化
        imputed_data = vm.renormalization(imputed_data_norm, self.norm_parameters)

        return imputed_data
