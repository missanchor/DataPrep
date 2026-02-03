import os
import pandas as pd
import numpy as np
from typing import Dict, Any, List
from dataprep.base import BaseEstimator

import dataprep.tabular.detection.ZeroED_modules as mo


class ZeroED(BaseEstimator):
    def __init__(self,
                 model_name="Pro/Qwen/Qwen2.5-7B-Instruct",
                 api_use=True,
                 base_url='https://api.siliconflow.cn/v1/',
                 local_model_use=True,
                 n_method='kmeans',
                 verbose=True,
                 result_dir='./result',
                 related_attrs=True,
                 distri_analysis=True,
                 guide_use=True,
                 err_gen_use=True,
                 rel_top=1,
                 func_val_threshold=0.6,
                 api_key=None,
                 **kwargs):

        # 1. 属性赋值
        self.model_name = model_name
        self.api_use = api_use
        self.base_url = base_url
        self.local_model_use = local_model_use
        self.n_method = n_method
        self.verbose = verbose
        self.result_dir = result_dir
        self.related_attrs = related_attrs
        self.distri_analysis = distri_analysis
        self.guide_use = guide_use
        self.err_gen_use = err_gen_use
        self.rel_top = rel_top
        self.func_val_threshold = func_val_threshold
        self.api_key = api_key

        # 2. 初始化状态容器
        self.local_models = {}  # 存储 sklearn 模型 {col: model}
        self.generated_funcs = {}  # 存储生成的函数代码 {col: [code]}
        self.is_trained_ = False

        # 3. 初始化日志
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        self.logger = mo.Logger(result_dir)

    def train(self, dirty_csv: pd.DataFrame, **kwargs):
        """
        训练主流程：
        在这里打包参数，并调用 modules 里的管线
        """
        self._create_temp_dir(prefix="zeroed_train_")

        self.logger.info(f"Starting ZeroED Fit on dataset shape: {dirty_csv.shape}...")

        # --- 打包参数 ---
        params = {
            'model_name': self.model_name,
            'api_use': self.api_use,
            'base_url':self.base_url,
            'local_model_use': self.local_model_use,
            'n_method': self.n_method,
            'verbose': self.verbose,
            'result_dir': self.result_dir,
            'related_attrs': self.related_attrs,
            'distri_analysis': self.distri_analysis,
            'guide_use': self.guide_use,
            'err_gen_use': self.err_gen_use,
            'rel_top': self.rel_top,
            'func_val_threshold': self.func_val_threshold,
            'api_key': self.api_key
        }

        # Phase 1: 计算全局相关性 (Global Correlation)
        related_attrs_map = mo.run_phase_1_correlation(dirty_csv, params, self.logger)

        # Main Loop: 对每一列进行 Pipeline 处理
        columns = dirty_csv.columns.tolist()
        for col_idx, attr_name in enumerate(columns):
            self.logger.info(f"=== Processing Column: {attr_name} ({col_idx + 1}/{len(columns)}) ===")

            # 调用核心管线：输入数据和参数，输出模型和规则函数
            model, funcs = mo.train_single_column_pipeline(
                attr_name=attr_name,
                df=dirty_csv,
                related_attrs_map=related_attrs_map,
                params=params,
                logger=self.logger
            )

            # 存储结果
            if model:
                self.local_models[attr_name] = model
            if funcs:
                self.generated_funcs[attr_name] = funcs

        self.is_trained_ = True
        self.logger.info("ZeroED Fit Complete.")

        self._save_checkpoint("zeroed_detector_complete.pkl")


    def predict(self, dirty_csv: pd.DataFrame) -> pd.DataFrame:
        """
        预测主流程
        """
        if not self.is_trained_:
            raise RuntimeError("Model is not trained. Run .train() first!")

        self.logger.info("Starting Prediction...")

        # 委托给 modules 处理整个 DataFrame 的预测逻辑
        result_mask = mo.predict_pipeline(
            dirty_csv,
            self.local_models,
            self.generated_funcs,
            self.local_model_use
        )

        return result_mask

    def train_and_predict(self, dirty_csv: pd.DataFrame) -> pd.DataFrame:
        self.train(dirty_csv)
        return self.predict(dirty_csv)