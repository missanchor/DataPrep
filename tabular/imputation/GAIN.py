import tensorflow as tf
import numpy as np
from tqdm import tqdm
import time
from dataprep.tabular.imputation import module
from dataprep.tabular.imputation.base import Imputation_Estimator


class GAIN(Imputation_Estimator):
    def __init__(self, batch_size=128, hint_rate=0.9, alpha=10,
                 epochs=100, epsilon=1.4, value=2,
                 gpu_device='0', **kwargs):
        self.params = {
            'batch_size': batch_size, 'hint_rate': hint_rate, 'alpha': alpha,
            'epochs': epochs, 'epsilon': epsilon, 'value': value,
            'gpu_device': gpu_device
        }
        self.is_trained_ = False
        self.sess = None
        self.norm_parameters = None
        self.graph = None
        # 保存序列化后的权重
        self.saved_weights_ = {}


    def train(self, data, missing_mask=None, **kwargs):
        start_time = time.time()

        # 1. 数据预处理
        data = np.array(data)
        no, dim = data.shape
        if missing_mask is None: missing_mask = 1 - np.isnan(data)

        norm_data, self.norm_parameters = module.normalization(data)
        norm_data_x = np.nan_to_num(norm_data, 0)
        data_m = np.array(missing_mask)

        # 2. 构建计算图
        self.graph = module.build_gain_graph(dim, self.params['batch_size'],
                                             self.params['epsilon'], self.params['value'],
                                             self.params['gpu_device'])

        # 3. 训练循环
        self.sess = tf.compat.v1.Session()
        self.sess.run(tf.compat.v1.global_variables_initializer())

        ph, ops = self.graph['ph'], self.graph['ops']

        pbar = tqdm(range(self.params['epochs']), desc="GAIN Training")
        for _ in pbar:
            idx_list = module.sample_batch_index(no, no)
            avg_mse = 0
            count = 0

            for i in range(0, len(idx_list), self.params['batch_size']):
                mb_idx = idx_list[i:i + self.params['batch_size']]
                if len(mb_idx) == 0: continue

                X_mb = norm_data_x[mb_idx, :]
                M_mb = data_m[mb_idx, :]
                Z_mb = module.uniform_sampler(0, 0.01, len(mb_idx), dim)
                H_mb = M_mb * module.binary_sampler(self.params['hint_rate'], len(mb_idx), dim)
                X_mb_in = M_mb * X_mb + (1 - M_mb) * Z_mb

                fd = {ph['X']: X_mb_in, ph['M']: M_mb, ph['H']: H_mb}

                self.sess.run(ops['D_solver'], feed_dict=fd)
                self.sess.run(ops['clip_d_op'])
                _, mse = self.sess.run([ops['G_solver'], self.graph['losses']['MSE']], feed_dict=fd)
                avg_mse += mse
                count += 1

            if count > 0: pbar.set_description(f"GAIN Epoch MSE: {avg_mse / count:.4f}")

        self.is_trained_ = True
        print(f"GAIN Training finished. Time: {time.time() - start_time:.2f}s")
        return self

    def predict(self, data, **kwargs):
        if not self.is_trained_:
            raise RuntimeError("This GAIN instance is not trained yet. Call 'train' with appropriate data before using 'predict'.")

        # 自动恢复 Session
        if self.sess is None:
            self._restore_session_from_weights(np.array(data).shape[1])

        data = np.array(data)
        no, dim = data.shape
        data_m = 1 - np.isnan(data)

        norm_data, _ = module.normalization(data, self.norm_parameters)
        norm_data_x = np.nan_to_num(norm_data, 0)

        Z_mb = module.uniform_sampler(0, 0.01, no, dim)
        X_mb = data_m * norm_data_x + (1 - data_m) * Z_mb

        imputed_norm = self.sess.run(self.graph['ops']['G_sample'],
                                     feed_dict={self.graph['ph']['X']: X_mb, self.graph['ph']['M']: data_m})[0]

        imputed_final = data_m * norm_data_x + (1 - data_m) * imputed_norm
        imputed_data = module.renormalization(imputed_final, self.norm_parameters)
        return module.rounding(imputed_data, data)

    # ==========================================
    # 序列化支持 (解决 joblib 保存 TF Session 问题)
    # ==========================================
    def _restore_session_from_weights(self, dim):
        """从保存的 numpy 权重重建 TF 图和 Session"""
        print("Restoring TensorFlow graph from saved weights...")
        # 1. 重建图结构
        self.graph = module.build_gain_graph(dim, self.params['batch_size'],
                                             self.params['epsilon'], self.params['value'],
                                             self.params['gpu_device'])
        # 2. 启动 Session
        self.sess = tf.compat.v1.Session()
        self.sess.run(tf.compat.v1.global_variables_initializer())

        # 3. 将保存的权重赋值回 TF 变量
        all_vars = tf.compat.v1.global_variables()
        assign_ops = []
        for v in all_vars:
            if v.name in self.saved_weights_:
                assign_ops.append(tf.compat.v1.assign(v, self.saved_weights_[v.name]))

        self.sess.run(assign_ops)