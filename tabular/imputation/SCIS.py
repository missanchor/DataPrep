import tensorflow as tf
import numpy as np
from tqdm import tqdm
import time
from dataprep.tabular.imputation import module
from dataprep.tabular.imputation.base import Imputation_Estimator


class SCIS(Imputation_Estimator):
    def __init__(self, batch_size=128, hint_rate=0.9, alpha=10,
                 epochs=1, guarantee=0.95, epsilon=1.4, value=2,
                 initial_num=20000, thre_value=0.001, gpu_device='2', **kwargs):
        self.params = {
            'batch_size': batch_size, 'hint_rate': hint_rate, 'alpha': alpha,
            'epochs': epochs, 'guarantee': guarantee, 'epsilon': epsilon, 'value': value,
            'initial_num': initial_num, 'thre_value': thre_value, 'gpu_device': gpu_device
        }
        self.sess = None
        self.is_trained_ = False
        self.norm_parameters = None
        self.graph = None
        self.saved_weights_ = {}  # 用于序列化

    def train(self, data, missing_mask=None, s_miss=1.0, **kwargs):
        """实现 BaseEstimator 的 train 接口"""
        start_time = time.time()

        all_data = np.array(data)
        total_no, dim = all_data.shape
        if missing_mask is None: missing_mask = 1 - np.isnan(all_data)

        all_norm_data, self.norm_parameters = module.normalization(all_data)

        # 数据子集划分
        subset_indices = np.random.randint(total_no, size=int(total_no * s_miss))
        working_data = all_data[subset_indices]
        working_mask = missing_mask[subset_indices]

        indices = np.random.randint(len(working_data), size=self.params['initial_num'] * 2)
        idx_init = indices[:self.params['initial_num']]
        idx_val = indices[self.params['initial_num']:]

        def prepare_subset(indices):
            d = working_data[indices]
            m = working_mask[indices]
            nd, _ = module.normalization(d)
            return np.nan_to_num(nd, 0), m

        data_init_x, mask_init = prepare_subset(idx_init)
        data_val_x, mask_val = prepare_subset(idx_val)

        # 构建图
        self.graph = module.build_scis_graph(dim, len(working_data), self.params['initial_num'],
                                             self.params['batch_size'], self.params['epsilon'],
                                             self.params['value'], self.params['thre_value'],
                                             self.params['gpu_device'])

        self.sess = tf.compat.v1.Session()
        self.sess.run(tf.compat.v1.global_variables_initializer())
        ph, ops = self.graph['ph'], self.graph['ops']

        # 三阶段训练
        self._train_loop(data_init_x, mask_init, ph, ops, "SCIS Phase 1")

        print("SCIS Phase 2: Searching Sample Size...")
        n_number = self._binary_search(data_val_x, mask_val, ph, len(working_data))
        print(f"SCIS: Estimated Sample Size = {n_number}")

        idx_final = np.random.randint(len(working_data), size=n_number)
        data_final_x, mask_final = prepare_subset(idx_final)
        self._train_loop(data_final_x, mask_final, ph, ops, "SCIS Phase 3")

        self.is_trained_ = True
        return self

    def predict(self, data, **kwargs):
        """实现 BaseEstimator 的 predict 接口"""
        if not self.is_trained_:
            raise RuntimeError("This SCIS instance is not trained yet. Call 'train' with appropriate data before using 'predict'.")

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

    # 辅助函数
    def _train_loop(self, data_x, data_m, ph, ops, desc):
        no, dim = data_x.shape
        pbar = tqdm(range(self.params['epochs']), desc=desc)
        for _ in pbar:
            idx_list = module.sample_batch_index(no, no)
            for i in range(0, len(idx_list), self.params['batch_size']):
                mb_idx = idx_list[i:i + self.params['batch_size']]
                if len(mb_idx) == 0: continue
                X_mb = data_x[mb_idx, :]
                M_mb = data_m[mb_idx, :]
                Z_mb = module.uniform_sampler(0, 0.01, len(mb_idx), dim)
                H_mb = M_mb * module.binary_sampler(self.params['hint_rate'], len(mb_idx), dim)
                X_mb_in = M_mb * X_mb + (1 - M_mb) * Z_mb
                fd = {ph['X']: X_mb_in, ph['M']: M_mb, ph['H']: H_mb}
                self.sess.run(ops['D_solver'], feed_dict=fd)
                self.sess.run(ops['clip_d_op'])
                self.sess.run(ops['G_solver'], feed_dict=fd)

    def _binary_search(self, data_x, data_m, ph, total_len):
        k = 20
        up, down = total_len, self.params['initial_num']
        median = int((up + down) / 2)
        dim = data_x.shape[1]
        idx_list = module.sample_batch_index(len(data_x), len(data_x))
        batches = [idx_list[i:i + 128] for i in range(0, len(idx_list), 128)]
        while median != down and median != up:
            success_count = 0
            for _ in range(k):
                diff = 0
                for mb_idx in batches:
                    if len(mb_idx) == 0: continue
                    X_mb = data_x[mb_idx];
                    M_mb = data_m[mb_idx]
                    Z_mb = module.uniform_sampler(0, 0.01, len(mb_idx), dim)
                    H_mb = M_mb * module.binary_sampler(self.params['hint_rate'], len(mb_idx), dim)
                    X_mb_in = M_mb * X_mb + (1 - M_mb) * Z_mb
                    fd = {ph['n_num']: median, ph['X']: X_mb_in, ph['M']: M_mb, ph['H']: H_mb}
                    N_loss, n_loss = self.sess.run([self.graph['scis']['N_RMSE'], self.graph['scis']['n_RMSE']],
                                                   feed_dict=fd)
                    diff += abs(n_loss - N_loss)
                if diff < self.params['thre_value']: success_count += 1
            if success_count > self.params['guarantee'] * k:
                up = median
            else:
                down = median
            median = int((up + down) / 2)
        return median

    # ==========================================
    # 序列化支持
    # ==========================================
    def _restore_session_from_weights(self, dim):
        print("Restoring SCIS from saved weights...")
        # 重建图
        self.graph = module.build_scis_graph(dim, 10000, self.params['initial_num'],
                                             self.params['batch_size'], self.params['epsilon'],
                                             self.params['value'], self.params['thre_value'],
                                             self.params['gpu_device'])
        self.sess = tf.compat.v1.Session()
        self.sess.run(tf.compat.v1.global_variables_initializer())
        all_vars = tf.compat.v1.global_variables()
        assign_ops = [tf.compat.v1.assign(v, self.saved_weights_[v.name])
                      for v in all_vars if v.name in self.saved_weights_]
        self.sess.run(assign_ops)