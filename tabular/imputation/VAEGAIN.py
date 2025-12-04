import tensorflow as tf
import numpy as np
import pandas as pd
from tqdm import tqdm
import time

from dataprep.tabular.imputation import module
from dataprep.base import BaseEstimator


class VAEGAIN(BaseEstimator):
    def __init__(self, batch_size=8, hint_rate=0.9, alpha=10, learning_rate=0.002,
                 epochs=5000,
                 encoder_h1=50, encoder_h2=20,
                 decoder_h1=50, decoder_h2=20,
                 latent_size=20,
                 gpu_device='1', **kwargs):
        """
        初始化 VAE-GAIN 模型
        """
        self.params = {
            'batch_size': batch_size,
            'hint_rate': hint_rate,
            'alpha': alpha,
            'learning_rate': learning_rate,
            'epochs': epochs,
            'encoder_h1': encoder_h1,
            'encoder_h2': encoder_h2,
            'decoder_h1': decoder_h1,
            'decoder_h2': decoder_h2,
            'latent_size': latent_size,
            'gpu_device': gpu_device
        }
        self.sess = None
        self.is_trained_ = False
        self.norm_parameters = None
        self.graph = None
        self.saved_weights_ = {}  # 用于序列化保存

    def train(self, data, missing_mask=None, **kwargs):
        """
        训练模型
        Args:
            data: DataFrame or Numpy Array, 含有缺失值
        """
        start_time = time.time()

        # 1. 数据处理 (Data Preparation)
        # 兼容 Pandas DataFrame
        if isinstance(data, pd.DataFrame):
            data = data.values

        data = np.array(data)
        # 处理字符串类型的 'nan' 如果存在
        if data.dtype.kind in {'U', 'S', 'O'}:
            data[data == 'nan'] = np.nan
            data = data.astype(float)

        no, dim = data.shape
        if missing_mask is None:
            missing_mask = 1 - np.isnan(data)
        else:
            missing_mask = np.array(missing_mask)

        # 归一化
        norm_data, self.norm_parameters = module.normalization(data)
        norm_data_x = np.nan_to_num(norm_data, 0)

        # 2. 构建计算图
        self.graph = module.build_vaegain_graph(dim, self.params, self.params['gpu_device'])

        # 3. 初始化会话
        gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
        self.sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))
        self.sess.run(tf.compat.v1.global_variables_initializer())

        ph = self.graph['ph']
        ops = self.graph['ops']
        losses = self.graph['losses']

        # 4. 训练循环
        # 注意：原代码的Epoch定义其实是Iteration (range(5000))
        # 这里的 epochs 参数对应原代码的迭代次数
        pbar = tqdm(range(self.params['epochs']), desc="VAE-GAIN Training")

        for _ in pbar:
            # Mini-batch sampling
            mb_idx = module.sample_batch_index(no, self.params['batch_size'])

            # Prepare Batch Data
            X_mb = norm_data_x[mb_idx, :]
            M_mb = missing_mask[mb_idx, :]
            Z_mb = module.uniform_sampler(0, 0.01, len(mb_idx), dim)

            # Hint Vector
            H_mb_raw = module.binary_sampler(1 - self.params['hint_rate'], len(mb_idx), dim)  # Original uses 1-p
            H_mb = M_mb * H_mb_raw

            # Introduce Noise/Missingness
            New_X_mb = M_mb * X_mb + (1 - M_mb) * Z_mb

            fd = {ph['M']: M_mb, ph['New_X']: New_X_mb, ph['H']: H_mb}

            # Update Discriminator
            _, d_loss_curr = self.sess.run([ops['D_solver'], losses['D_loss']], feed_dict=fd)

            # Update VAE (Generator)
            _, vae_loss_curr = self.sess.run([ops['VAE_solver'], losses['VAE_loss']], feed_dict=fd)

            # Optional: Print loss occasionally
            # pbar.set_description(f"D_loss: {d_loss_curr:.4f}, VAE_loss: {vae_loss_curr:.4f}")

        self.is_trained_ = True
        print(f"VAE-GAIN Training finished. Time: {time.time() - start_time:.2f}s")
        return self

    def predict(self, data, **kwargs):
        """
        预测 (补全) 数据
        """
        if not self.is_trained_:
            raise RuntimeError("Train first.")

        # 自动恢复 Session (如果在加载模型后)
        if self.sess is None:
            data_dim = np.array(data).shape[1]
            self._restore_session_from_weights(data_dim)

        # 1. 数据准备
        if isinstance(data, pd.DataFrame):
            original_columns = data.columns
            data = data.values
        else:
            original_columns = None

        data = np.array(data)
        if data.dtype.kind in {'U', 'S', 'O'}:
            data[data == 'nan'] = np.nan
            data = data.astype(float)

        no, dim = data.shape
        data_m = 1 - np.isnan(data)

        # 归一化 (使用训练时的参数)
        norm_data, _ = module.normalization(data, self.norm_parameters)
        norm_data_x = np.nan_to_num(norm_data, 0)

        # 2. 全量生成
        # 原逻辑：New_Data_mb = M_mb * X_mb + (1 - M_mb) * Data_Z_mb
        Data_Z_mb = module.uniform_sampler(0, 0.01, no, dim)
        New_Data_mb = data_m * norm_data_x + (1 - data_m) * Data_Z_mb

        ph = self.graph['ph']
        ops = self.graph['ops']

        # 生成补全值
        imputed_norm = self.sess.run(ops['G_sample'],
                                     feed_dict={ph['New_X']: New_Data_mb, ph['M']: data_m})

        # 3. 结果融合与反归一化
        imputed_final = data_m * norm_data_x + (1 - data_m) * imputed_norm
        imputed_data = module.renormalization(imputed_final, self.norm_parameters)
        imputed_data = module.rounding(imputed_data, data)

        return imputed_data

    # ==========================================
    # 序列化支持 (BaseEstimator 兼容)
    # ==========================================
    def __getstate__(self):
        state = self.__dict__.copy()
        if self.sess is not None:
            vars_list = tf.compat.v1.global_variables()
            vars_vals = self.sess.run(vars_list)
            state['saved_weights_'] = {v.name: val for v, val in zip(vars_list, vars_vals)}
        if 'sess' in state: del state['sess']
        if 'graph' in state: del state['graph']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.sess = None
        self.graph = None

    def _restore_session_from_weights(self, dim):
        print("Restoring VAE-GAIN from saved weights...")
        self.graph = module.build_vaegain_graph(dim, self.params, self.params['gpu_device'])
        gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
        self.sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))
        self.sess.run(tf.compat.v1.global_variables_initializer())

        all_vars = tf.compat.v1.global_variables()
        assign_ops = [tf.compat.v1.assign(v, self.saved_weights_[v.name])
                      for v in all_vars if v.name in self.saved_weights_]
        self.sess.run(assign_ops)