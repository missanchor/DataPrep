# dataprep/correction/base.py

import os
import abc
import time
import json
import pandas as pd
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import ast # 确保导入 ast
from collections import Counter # 确保导入 Counter


class BaseDataCorrector(abc.ABC):
    def __init__(self, output_path: str = "./outputs"):
        self.output_path = output_path
        os.makedirs(output_path, exist_ok=True)
        self.logs: List[Dict[str, Any]] = []
        # _dirty_data_ref 用于在 evaluate 方法中引用原始脏数据，必须在 fit 中设置
        self._dirty_data_ref: Optional[pd.DataFrame] = None
        print(f"BaseDataCorrector initialized with output path: {self.output_path}")

    # 这就是关键！_safe_equals 必须作为 BaseDataCorrector 的方法而存在。
    def _safe_equals(self, val1, val2) -> bool:
        # Handle NaN comparisons
        if pd.isna(val1) and pd.isna(val2):
            return True
        if pd.isna(val1) or pd.isna(val2):
            return False

        # Convert non-string types to string for robust comparison, especially for mixed types
        str_val1 = str(val1).strip()
        str_val2 = str(val2).strip()

        # Special handling for numerical strings that might be represented differently
        try:
            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                return np.isclose(val1, val2, rtol=1e-09, atol=1e-09, equal_nan=True)
            # Try converting to float if both are numeric-like strings
            float_val1 = float(str_val1)
            float_val2 = float(str_val2)
            return np.isclose(float_val1, float_val2, rtol=1e-09, atol=1e-09, equal_nan=True)
        except ValueError:
            pass # Not numerical, fall through to string comparison

        # Finally, perform case-insensitive string comparison for all other types
        return str_val1.lower() == str_val2.lower()

    @abc.abstractmethod
    def fit(self, dirty_data: pd.DataFrame, clean_data_for_learning: Optional[pd.DataFrame] = None, **kwargs):
        """
        训练或配置数据纠正器。
        对于监督式模型，这可能涉及从 clean_data_for_learning 中学习。
        对于零样本模型，这可能只是设置上下文。
        此方法**必须**将 `dirty_data` 的副本存储到 `self._dirty_data_ref`，
        以便 `evaluate` 方法可以访问原始的脏数据。
        """
        raise NotImplementedError

    @abc.abstractmethod
    def predict(self, data_to_correct: pd.DataFrame) -> pd.DataFrame:
        """
        根据训练好的模型或配置对数据进行纠正。
        """
        raise NotImplementedError

    def evaluate(self, dirty_data: pd.DataFrame, clean_data: pd.DataFrame, corrected_data: pd.DataFrame) -> Dict[
        str, float]:
        """
        评估纠正器的性能。
        参数:
            clean_data (pd.DataFrame): 干净（ground truth）的数据。
            corrected_data (pd.DataFrame): 纠正后的数据。
        返回:
            Dict[str, float]: 包含 Precision, Recall, F1-Score 的字典。
        """
        if self._dirty_data_ref is None:
            raise ValueError("'_dirty_data_ref' must be set by the 'fit' method before calling 'evaluate'.")

        true_positives = 0
        false_positives = 0
        false_negatives = 0

        # 确保所有 DataFrame 的索引和列名一致
        if not self._dirty_data_ref.index.equals(clean_data.index) or \
           not self._dirty_data_ref.columns.equals(clean_data.columns) or \
           not corrected_data.index.equals(clean_data.index) or \
           not corrected_data.columns.equals(clean_data.columns):
            raise ValueError("Dirty, clean, and corrected dataframes must have matching indices and columns.")

        for r_idx in clean_data.index:
            for col_name in clean_data.columns:
                dirty_value = self._dirty_data_ref.at[r_idx, col_name]
                gt_value = clean_data.at[r_idx, col_name]
                corrected_value = corrected_data.at[r_idx, col_name]

                # **重要：确保这里以及 ZeroEC 中所有用到 safe_equals 的地方，都改成了 self._safe_equals**
                is_gt_clean = self._safe_equals(dirty_value, gt_value)
                is_corrected_clean = self._safe_equals(corrected_value, gt_value)
                is_algorithm_changed = not self._safe_equals(dirty_value, corrected_value)

                if not is_gt_clean:  # 原始数据是脏的
                    if is_algorithm_changed:  # 算法尝试纠正了它
                        if is_corrected_clean:
                            true_positives += 1  # 成功纠正，TP
                        else:
                            false_positives += 1  # 纠正了但错了，FP (也可以视为新的错误)
                    else:  # 算法没有尝试纠正脏数据
                        false_negatives += 1  # 漏报，FN
                else:  # 原始数据是干净的
                    if is_algorithm_changed:  # 算法改变了干净数据
                        false_positives += 1  # 误报，FP

        # 计算 Precision, Recall, F1-Score
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        return {
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1_score
        }

    def save(self, path: str):
        """
        保存纠正器模型的状态。
        """
        raise NotImplementedError

    @classmethod
    def load(cls, path: str, llm_config: Optional[Dict[str, Any]] = None):
        """
        加载纠正器模型的状态。
        """
        raise NotImplementedError

    def save_print_logs(self):
        """
        将当前运行的日志保存到文件并打印。
        """
        if not self.logs:
            print("No logs to save or print for the current run.")
            return

        log_file_path = os.path.join(self.output_path, "correction_log.json")
        try:
            with open(log_file_path, 'w', encoding='utf-8') as f:
                json.dump(self.logs, f, ensure_ascii=False, indent=4)
            print(f"Correction logs saved to: {log_file_path}")
            # 打印日志内容
            print("\n--- Current Correction Logs ---")
            for log_entry in self.logs:
                print(json.dumps(log_entry, ensure_ascii=False, indent=2))
            print("--- End of Logs ---\n")
        except Exception as e:
            print(f"Error saving/printing logs: {e}")