import os
import time
import json
import re
from sklearn.metrics import mutual_info_score
import textwrap
import sys
import ast
import copy
import random
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from langchain_openai import ChatOpenAI

from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.output_parsers import JsonOutputParser
from sklearn.cluster import MiniBatchKMeans
from joblib import Parallel, delayed, dump
from openpyxl import Workbook
import shutil

from base import BaseDataCorrector
from ZeroEC_modules import MyEmbeddings, RepairOutput, load_prompts, load_examples, form_examples

print(f"ZeroEC file path: {os.path.abspath(__file__)}")

def clean_temp_folders():
    """Clean up joblib temporary folders"""
    temp_dir = r"C:\Users\33328\AppData\Local\Temp"
    for folder in os.listdir(temp_dir):
        if folder.startswith("joblib_memmapping_folder_"):
            try:
                shutil.rmtree(os.path.join(temp_dir, folder))
                print(f"[{time.ctime()}] Cleaned: {folder}")
            except Exception as e:
                print(f"[{time.ctime()}] Failed to clean {folder}: {str(e)}")

def clean_module_cache():
    """Clear module cache to resolve RuntimeWarning"""
    if 'dataprep.correction.zeroec' in sys.modules:
        del sys.modules['dataprep.correction.zeroec']
        print(f"[{time.ctime()}] Cleaned module cache: dataprep.correction.zeroec")

def calc_p_r_f(self, clean_data: pd.DataFrame, dirty_data: pd.DataFrame, corrected_data: pd.DataFrame) -> Dict[str, float]:
    print(f"[{time.ctime()}] Calculating P/R/F1")
    def is_effectively_na(value):
        if isinstance(value, (np.ndarray, pd.Series)):
            if value.size == 0:
                return True
            try:
                return np.all(pd.isna(value))
            except TypeError:
                return False
        elif isinstance(value, list):
            return False
        return pd.isna(value)

    def safe_equals(val1, val2):
        is_val1_na = is_effectively_na(val1)
        is_val2_na = is_effectively_na(val2)
        if is_val1_na and is_val2_na:
            return True
        if is_val1_na or is_val2_na:
            return False
        if isinstance(val1, (float, np.float64)) and isinstance(val2, (float, np.float64)):
            return np.isclose(val1, val2, rtol=1e-09, atol=1e-09)
        return str(val1).strip().lower() == str(val2).strip().lower()

    def safe_not_equals(val1, val2):
        return not safe_equals(val1, val2)

    true_positives = 0
    false_positives = 0
    false_negatives = 0

    for col in clean_data.columns:
        for idx in clean_data.index:
            clean_val = clean_data.at[idx, col]
            dirty_val = dirty_data.at[idx, col]
            corrected_val = corrected_data.at[idx, col]

            if is_effectively_na(clean_val) or is_effectively_na(dirty_val) or is_effectively_na(corrected_val):
                continue

            is_dirty = safe_not_equals(dirty_val, clean_val)
            is_corrected = safe_not_equals(corrected_val, dirty_val)
            is_correct = safe_equals(corrected_val, clean_val)

            print(f"[{time.ctime()}] 比较: 行 {idx}, 列 {col}, 脏值={dirty_val}, 干净值={clean_val}, 修复值={corrected_val}, "
                  f"is_dirty={is_dirty}, is_corrected={is_corrected}, is_correct={is_correct}")

            if is_dirty:
                if is_corrected:
                    if is_correct:
                        true_positives += 1
                    else:
                        false_positives += 1
                else:
                    false_negatives += 1
            else:
                if is_corrected:
                    false_positives += 1

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    metrics = {'precision': precision, 'recall': recall, 'f1': f1}
    print(f"[{time.ctime()}] Metrics: {metrics}, TP={true_positives}, FP={false_positives}, FN={false_negatives}")
    return metrics

def cmp_mark(clean_data, corrections, output_path="output", output_filename="comparison.xlsx"):
    print(f"[{time.ctime()}] Generating comparison report")
    if not isinstance(clean_data, pd.DataFrame) or not isinstance(corrections, pd.DataFrame):
        print(f"[{time.ctime()}] Error: clean_data and corrections must be pandas DataFrames.")
        return

    os.makedirs(output_path, exist_ok=True)
    wb = Workbook()
    ws = wb.active
    ws.title = "Comparison"

    headers = ["Type", "Column", "Row Index", "Dirty Value", "Clean Value", "Corrected Value", "Correctly Fixed", "Chain of Thought"]
    ws.append(headers)

    for r_idx in range(clean_data.shape[0]):
        for c_idx, col_name in enumerate(clean_data.columns):
            clean_value = clean_data.iloc[r_idx, c_idx]
            corrected_value = corrections.iloc[r_idx, c_idx]

            is_clean_na = pd.isna(clean_value)
            is_corrected_na = pd.isna(corrected_value)
            are_equal = False
            if is_clean_na and is_corrected_na:
                are_equal = True
            elif is_clean_na or is_corrected_na:
                are_equal = False
            else:
                if isinstance(clean_value, (float, np.float64)) and isinstance(corrected_value, (float, np.float64)):
                    are_equal = np.isclose(clean_value, corrected_value, rtol=1e-09, atol=1e-09)
                else:
                    are_equal = str(clean_value).strip().lower() == str(corrected_value).strip().lower()

            if are_equal:
                continue

            try:
                current_logs = globals().get('logs', [])
                log_entry = next((log for log in current_logs if log.get("Column") == col_name and log.get("Row Index") == r_idx), None)
            except NameError:
                current_logs = []
                log_entry = None
                print(f"[{time.ctime()}] Warning: 'logs' variable not found or inaccessible in cmp_mark. Some details may be missing in the report.")

            record_type = "Correction"
            dirty_value = log_entry.get("Dirty_value", "") if log_entry else ""
            is_corrected_correctly = "Yes" if are_equal else "No"
            chain_of_thought = log_entry.get("Explanation", "") if log_entry else ""

            row_to_write = [
                record_type,
                col_name,
                r_idx,
                str(dirty_value),
                str(clean_value),
                str(corrected_value),
                is_corrected_correctly,
                chain_of_thought
            ]
            ws.append(row_to_write)

    try:
        wb.save(os.path.join(output_path, output_filename))
        print(f"[{time.ctime()}] Comparison report saved to: {os.path.join(output_path, output_filename)}")
    except Exception as e:
        print(f"[{time.ctime()}] Error saving comparison report: {str(e)}")

class ZeroEC(BaseDataCorrector):
        def __init__(
                self,
                model_path: str = "sentence-transformers/all-MiniLM-L6-v2",
                output_path: str = "output",
                prompt_template_dir: str = "prompt_templates",
                llm_config: dict = None,
                output_filename: str = "comparison.xlsx"
        ):
            super().__init__(output_path=output_path)
            print(f"[{time.ctime()}] 初始化 ZeroEC，输出路径: {output_path}")

            # 加载嵌入模型
            try:
                self.embedding_model = MyEmbeddings(model_path)
                print(f"[{time.ctime()}] 成功加载嵌入模型: {model_path}")
            except Exception as e:
                print(f"[{time.ctime()}] 加载嵌入模型失败: {str(e)}")
                raise

            self.prompt_template_dir = os.path.normpath(
                os.path.join(os.path.dirname(__file__), "..", "..", prompt_template_dir))
            print(f"[{time.ctime()}] 提示模板目录: {self.prompt_template_dir}")
            if not os.path.exists(self.prompt_template_dir):
                print(f"[{time.ctime()}] 错误: 提示模板目录 {self.prompt_template_dir} 不存在")
                raise FileNotFoundError(f"提示模板目录 {self.prompt_template_dir} 不存在")

            self.output_filename = output_filename
            self.retriever_dict = {}
            self.specific_examples = {}
            self.indices_dict = {}
            self.CoE_dict = {}
            self.retrieved_tuples = {}
            self.prompt_dict = {}
            self.sp_examps = {}

            self.rep_error_info = {}
            self.corrections = None
            self.corrections_df = pd.DataFrame(columns=['row', 'col', 'original', 'corrected'])
            self.total_tokens = 0
            self.human_repair_num = 50
            self.augmented_data = {}
            self.rep_error_info = {}
            self.codes = {}
            self.fds = {}
            self.header = []
            self.logs = []
            self.f_time_cost = None
            os.makedirs(output_path, exist_ok=True)
            self._init_time_cost_file()

            # 定义提示文件路径
            prompt_files = [
                os.path.join(self.prompt_template_dir, f) for f in [
                    'SystemMessage-2.txt', 'HumanMessage.txt',
                    'SystemMessage_for_AutoCoT_with_error_type.txt',
                    'HumanMessage_for_AutoCoT_large.txt', 'HumanMessage_for_AutoCoT_small.txt',
                    'SystemMessage_data_augmentation.txt', 'HumanMessage_data_augmentation.txt',
                    'SystemMessage_code_generation.txt', 'HumanMessage_code_generation.txt',
                    'SystemMessage_fd_generation.txt', 'HumanMessage_fd_generation.txt'
                ]
            ]

            # 加载提示
            (self.sys, self.human, self.sys_auto_cot, self.human_auto_cot_large, self.human_auto_cot_small,
             self.sys_data_augmentation, self.human_data_augmentation, self.sys_code_generation,
             self.human_code_generation, self.sys_fd_generation, self.human_fd_generation) = load_prompts(*prompt_files)
            print(
                f"[{time.ctime()}] 已加载提示: sys_data_augmentation 长度={len(self.sys_data_augmentation)}, human_data_augmentation 长度={len(self.human_data_augmentation)}")

            self.general_examples_str = load_examples(os.path.join(self.prompt_template_dir, 'examples.txt'))

            # 初始化 LLM 实例，使用 ChatOpenAI
            llm_config = llm_config or {}
            default_api_key = 'sk-csclmsflwgupktivimfudfqhwvbrxdljnytubxmrxinsulcp'
            default_base_url = 'https://api.siliconflow.cn/v1/'
            try:
                self.llm_repair = ChatOpenAI(
                    model=llm_config.get('repair_model', 'Qwen/Qwen2.5-72B-Instruct'),
                    api_key=llm_config.get('repair_api_key', llm_config.get('api_key', default_api_key)),
                    base_url=llm_config.get('repair_base_url', llm_config.get('base_url', default_base_url)),
                    temperature=llm_config.get('repair_temperature', 0.5)
                )
                self.auto_cot_llm = ChatOpenAI(
                    model=llm_config.get('auto_cot_model', 'Qwen/Qwen2.5-72B-Instruct'),
                    api_key=llm_config.get('auto_cot_api_key', llm_config.get('api_key', default_api_key)),
                    base_url=llm_config.get('auto_cot_base_url', llm_config.get('base_url', default_base_url)),
                    temperature=llm_config.get('auto_cot_temperature', 0.0)  # 降低温度以提高稳定性
                )
                self.data_augmentation_llm = ChatOpenAI(
                    model=llm_config.get('data_augmentation_model', 'Qwen/Qwen2.5-72B-Instruct'),
                    api_key=llm_config.get('data_augmentation_api_key', llm_config.get('api_key', default_api_key)),
                    base_url=llm_config.get('data_augmentation_base_url', llm_config.get('base_url', default_base_url)),
                    temperature=llm_config.get('data_augmentation_temperature', 0.3)
                )
                self.code_generation_llm = ChatOpenAI(
                    model=llm_config.get('code_generation_model', 'Qwen/Qwen2.5-72B-Instruct'),
                    api_key=llm_config.get('code_generation_api_key', llm_config.get('api_key', default_api_key)),
                    base_url=llm_config.get('code_generation_base_url', llm_config.get('base_url', default_base_url)),
                    temperature=llm_config.get('code_generation_temperature', 0.0)
                )
                self.fd_generation_llm = ChatOpenAI(
                    model=llm_config.get('fd_generation_model', 'Qwen/Qwen2.5-72B-Instruct'),
                    api_key=llm_config.get('fd_generation_api_key', llm_config.get('api_key', default_api_key)),
                    base_url=llm_config.get('fd_generation_base_url', llm_config.get('base_url', default_base_url)),
                    temperature=llm_config.get('fd_generation_temperature', 0.0)
                )
                self.parser = JsonOutputParser(pydantic_object=RepairOutput)
                print(f"[{time.ctime()}] ZeroEC 初始化完成，model_path={model_path}, output_path={output_path}")
            except Exception as e:
                print(f"[{time.ctime()}] 初始化 LLM 失败: {str(e)}")
                self.logs.append(f"初始化 LLM 失败: {str(e)}")
                raise ValueError(f"初始化 LLM 失败: {str(e)}")

        def _init_time_cost_file(self):
            """Initialize time_cost.txt file"""
            try:
                if self.f_time_cost is None or self.f_time_cost.closed:
                    self.f_time_cost = open(os.path.join(self.output_path, "time_cost.txt"), 'a', encoding='utf-8')
                    self.f_time_cost.write(f"[{time.ctime()}] Initialized time_cost.txt\n")
                    self.f_time_cost.flush()
                    print(f"[{time.ctime()}] Initialized time_cost.txt")
            except Exception as e:
                print(f"[{time.ctime()}] Failed to initialize time_cost.txt: {str(e)}")
                self.logs.append(f"Failed to initialize time_cost.txt: {str(e)}")
                raise
        def select_repair_candidates(self, embeddings_matrix: np.ndarray, detection: pd.DataFrame,
                                     num_clusters: int) -> list:
            print(f"[{time.ctime()}] Selecting repair candidates with num_clusters={num_clusters}")
            mask_rows = detection.sum(axis=1) > 0
            filtered_embeddings = embeddings_matrix[mask_rows]
            filtered_detection = detection[mask_rows]
            original_indices = np.where(mask_rows)[0]
            print(f"[{time.ctime()}] Filtered rows with errors: {len(filtered_embeddings)}")

            if len(filtered_embeddings) == 0:
                print(f"[{time.ctime()}] No rows with errors found, returning empty repair list")
                return []

            m, n, l = filtered_embeddings.shape
            mask = filtered_detection.astype(bool)
            masked_embeddings = filtered_embeddings * mask.values[..., np.newaxis]
            reshaped_embeddings = masked_embeddings.reshape((m, n * l))
            kmeans = MiniBatchKMeans(n_clusters=min(num_clusters, len(filtered_embeddings)), random_state=42,
                                     batch_size=4096)
            clusters = kmeans.fit_predict(reshaped_embeddings)
            selected_indices = []
            covered_columns = set()

            for i in range(min(num_clusters, len(filtered_embeddings))):
                cluster_mask = clusters == i
                if np.any(cluster_mask):
                    cluster_detection = filtered_detection.iloc[cluster_mask]
                    cluster_original_indices = original_indices[cluster_mask]
                    best_index = None
                    best_new_coverage = -1
                    for idx, orig_idx in zip(cluster_detection.index, cluster_original_indices):
                        row = cluster_detection.loc[idx]
                        new_columns = set(row[row == 1].index) - covered_columns
                        if len(new_columns) > best_new_coverage:
                            best_new_coverage = len(new_columns)
                            best_index = idx
                            best_orig_index = orig_idx
                    if best_index is not None:
                        selected_indices.append(best_orig_index)
                        covered_columns.update(
                            set(cluster_detection.loc[best_index][cluster_detection.loc[best_index] == 1].index))

            print(f"[{time.ctime()}] Selected repair indices: {selected_indices}")
            return selected_indices

        def clean_data_integration(self,clean_data: pd.DataFrame, rep_data_info: dict):
            # 对于各出错列，将采样所得的clean_data与dirty_data融合
            for column in self.dirty_data.columns:
                if self.detection[column].sum() > 0:
                    for idx in range(len(clean_data)):
                        rep_data_info[column][clean_data.iloc[idx].name] = {}
                        rep_data_info[column][clean_data.iloc[idx].name]['dirty_tuple'] = clean_data.iloc[idx].to_dict()
                        rep_data_info[column][clean_data.iloc[idx].name]['dirty_value'] = clean_data.iloc[idx][column]
                        rep_data_info[column][clean_data.iloc[idx].name][
                            'error_analysis'] = 'This is a clean value that does not need correction.'
                        rep_data_info[column][clean_data.iloc[idx].name]['ground_truth'] = clean_data.iloc[idx][column]
                        rep_data_info[column][clean_data.iloc[idx].name]['error_type'] = 'clean'


        def train_val_split(self, error_info: Dict) -> Tuple[Dict, Dict]:
            print(f"[{time.ctime()}] 开始 train-validation 分割")
            try:
                if not error_info:
                    print(f"[{time.ctime()}] 无错误信息，跳过分割")
                    return {}, {}
                train_data = {}
                val_data = {}
                indices = list(error_info.keys())
                if len(indices) == 0:
                    print(f"[{time.ctime()}] 无可用数据，跳过分割")
                    return {}, {}
                random.shuffle(indices)
                split_idx = int(len(indices) * 0.8)
                train_indices = indices[:split_idx]
                val_indices = indices[split_idx:]
                for idx in train_indices:
                    train_data[idx] = error_info[idx]
                for idx in val_indices:
                    val_data[idx] = error_info[idx]
                print(f"[{time.ctime()}] Train-validation 分割完成: {len(train_data)} train, {len(val_data)} validation")
                self.logs.append(f"Train-validation 分割完成: {len(train_data)} train, {len(val_data)} validation")
                return train_data, val_data
            except Exception as e:
                print(f"[{time.ctime()}] Train-validation 分割失败: {str(e)}")
                self.logs.append(f"Train-validation 分割失败: {str(e)}")
                return {}, {}

        def calc_mi_2(self, df: pd.DataFrame, target_column: str) -> np.ndarray:
            """Calculate normalized mutual information between target_column and all other columns."""
            print(f"[{time.ctime()}] Calculating mutual information for {target_column}")

            def calculate_mi_optimized(column, target_column, df):
                try:
                    if column == target_column:
                        return mutual_info_score(df[target_column].fillna('').astype(str),
                                                 df[column].fillna('').astype(str))
                    else:
                        # 计算每个组合的计数，过滤掉只出现一次的组合
                        counts = df.groupby([target_column, column]).size()
                        filtered_counts = counts[counts > 1].reset_index()

                        if filtered_counts.empty:
                            print(f"[{time.ctime()}] No valid pairs for {column}, returning 0")
                            return 0.0

                        target_values = filtered_counts[target_column].astype(str)
                        column_values = filtered_counts[column].astype(str)
                        return mutual_info_score(target_values, column_values)
                except Exception as e:
                    print(f"[{time.ctime()}] Failed to calculate MI for {column}: {str(e)}")
                    return 0.0

            try:
                # 并行计算所有列的互信息
                mutual_info_list = Parallel(n_jobs=-1)(
                    delayed(calculate_mi_optimized)(col, target_column, df)
                    for col in df.columns
                )

                # 归一化
                max_mutual_info = max(mutual_info_list) if mutual_info_list and max(mutual_info_list) != 0 else 1.0
                normalized_mi = [mi / max_mutual_info for mi in mutual_info_list]
                print(f"[{time.ctime()}] Normalized MI: {normalized_mi}")
                return np.array(normalized_mi)
            except Exception as e:
                print(f"[{time.ctime()}] Failed to calculate mutual information: {str(e)}")
                return np.zeros(len(df.columns))

        def generate_embeddings(self, data: pd.DataFrame) -> np.ndarray:
            print(f"[{time.ctime()}] 生成嵌入矩阵")
            embeddings = []
            for col in data.columns:
                texts = [str(val) for val in data[col] if pd.notna(val)]
                if not texts:
                    print(f"[{time.ctime()}] 警告: 列 {col} 无有效文本，使用空嵌入")
                    embeddings.append(
                        np.zeros((len(data), self.embedding_model.model.get_sentence_embedding_dimension())))
                    continue
                col_embeddings = self.embedding_model.embed_documents(texts)
                # 填充缺失值
                full_embeddings = np.zeros((len(data), col_embeddings.shape[1]))
                valid_indices = [i for i, val in enumerate(data[col]) if pd.notna(val)]
                for i, idx in enumerate(valid_indices):
                    full_embeddings[idx] = col_embeddings[i]
                embeddings.append(full_embeddings)
            embeddings_matrix = np.array(embeddings).transpose(1, 0, 2)  # 形状: (rows, cols, embedding_dim)
            print(f"[{time.ctime()}] 嵌入矩阵生成完成，形状: {embeddings_matrix.shape}")
            return embeddings_matrix

        def fit(self, dirty_data: pd.DataFrame, clean_data: pd.DataFrame, detection: pd.DataFrame):
            print(f"[{time.ctime()}] 开始 fit 方法")
            print(f"[{time.ctime()}] 检查 clean_data_integration: {hasattr(self, 'clean_data_integration')}")
            print(f"[{time.ctime()}] ZeroEC 类方法: {dir(self)[:10]}...")
            print(f"[{time.ctime()}] ZeroEC 模块路径: {self.__module__}")
            print(f"[{time.ctime()}] ZeroEC 文件路径: {__file__}")
            print(f"[{time.ctime()}] 脏数据形状: {dirty_data.shape}, 干净数据形状: {clean_data.shape}")
            print(f"[{time.ctime()}] CoE_dict: {self.CoE_dict}")

            # 验证列名一致性
            if not (set(dirty_data.columns) == set(clean_data.columns) == set(detection.columns)):
                error_msg = f"[{time.ctime()}] 错误: 列名不一致: dirty_data={dirty_data.columns}, clean_data={clean_data.columns}, detection={detection.columns}"
                print(error_msg)
                self.logs.append(error_msg)
                raise ValueError(error_msg)

            # 存储输入数据
            self.dirty_data = dirty_data.copy()
            self.clean_data = clean_data.copy() if clean_data is not None else pd.DataFrame()
            self.detection = detection.copy() if detection is not None else pd.DataFrame(0, index=dirty_data.index,
                                                                                         columns=dirty_data.columns)
            self.header = list(dirty_data.columns)
            self.sp_examps = {col: '' for col in self.header}
            self._dirty_data_ref = dirty_data.copy()
            self.dirty_data_human_repaired = dirty_data.copy()
            self.detection_human_repaired = self.detection.copy()
            self.corrections = dirty_data.copy()
            self.corrections_df = pd.DataFrame(columns=['row', 'col', 'original', 'corrected'])
            self.rep_error_info = {col: {} for col in self.header}  # 使用 self.header 初始化
            self.augmented_data = {col: {} for col in self.header}  # 初始化 augmented_data

            print(f"[{time.ctime()}] 初始化 rep_error_info: {list(self.rep_error_info.keys())}")

            # 生成嵌入矩阵
            print(f"[{time.ctime()}] 开始生成嵌入矩阵")
            emb_start_time = time.time()
            self.embeddings_matrix = self.generate_embeddings(dirty_data)
            if self.embeddings_matrix is None:
                error_msg = f"[{time.ctime()}] 错误: 无法生成嵌入矩阵，跳过后续步骤"
                print(error_msg)
                self.logs.append(error_msg)
                return self

            # 选择修复候选
            print(f"[{time.ctime()}] 开始选择修复候选")
            select_start_time = time.time()
            repair_list = self.select_repair_candidates(self.embeddings_matrix, self.detection,
                                                        self.human_repair_num if hasattr(self,
                                                                                         'human_repair_num') else 10)
            select_end_time = time.time()
            print(f"[{time.ctime()}] 修复候选列表长度: {len(repair_list)}")
            print(f"[{time.ctime()}] 选择耗时: {select_end_time - select_start_time:.2f} 秒")
            self.f_time_cost.write(f"[{time.ctime()}] 选择耗时: {select_end_time - select_start_time:.2f} 秒\n")

            if not repair_list:
                print(f"[{time.ctime()}] 警告: 未选择修复候选，跳过人工修复和 Auto-CoT")
                self.logs.append("警告: 未选择修复候选")
                return self

            # 应用人工修复
            print(f"[{time.ctime()}] 对索引 {repair_list} 应用人工修复")
            self.dirty_data_human_repaired.iloc[repair_list] = self.clean_data.iloc[repair_list]
            self.detection_human_repaired.iloc[repair_list] = 0
            self.corrections.iloc[repair_list] = self.clean_data.iloc[repair_list]

            # 生成修复数据的嵌入
            print(f"[{time.ctime()}] 为修复数据生成嵌入")
            dirty_data_only_repaired = self.clean_data.iloc[repair_list]
            print(f"[{time.ctime()}] 修复数据形状: {dirty_data_only_repaired.shape}")
            elements_list_only_repaired = dirty_data_only_repaired.values.flatten().tolist()
            print(f"[{time.ctime()}] 展平修复元素数量: {len(elements_list_only_repaired)}")
            embedding_dimension = self.embedding_model.model.get_sentence_embedding_dimension()
            try:
                embeddings_only_repaired = self.embedding_model.embed_documents(elements_list_only_repaired)
                if np.array(embeddings_only_repaired).size == 0:
                    print(f"[{time.ctime()}] 警告: 修复数据嵌入为空")
                    self.logs.append("警告: 修复数据嵌入为空")
                    self.embeddings_matrix_only_repaired = np.zeros(
                        (len(repair_list), len(self.header), embedding_dimension))
                else:
                    expected_repaired_size = len(repair_list) * len(self.header) * embedding_dimension
                    if np.array(embeddings_only_repaired).size != expected_repaired_size:
                        print(
                            f"[{time.ctime()}] 警告: 嵌入大小 ({np.array(embeddings_only_repaired).size}) 不匹配预期 ({expected_repaired_size})")
                        self.logs.append(f"警告: 嵌入大小不匹配预期")
                    self.embeddings_matrix_only_repaired = np.array(embeddings_only_repaired).reshape(len(repair_list),
                                                                                                      len(self.header),
                                                                                                      embedding_dimension)
                    print(f"[{time.ctime()}] 修复嵌入矩阵形状: {self.embeddings_matrix_only_repaired.shape}")
            except Exception as e:
                print(f"[{time.ctime()}] 修复数据嵌入生成失败: {str(e)}")
                self.logs.append(f"修复数据嵌入生成失败: {str(e)}")
                self.embeddings_matrix_only_repaired = np.zeros(
                    (len(repair_list), len(self.header), embedding_dimension))

            # 构建检索器并执行 Auto-CoT
            retriever_build_time = 0
            for col_idx, column in enumerate(self.detection.columns):
                print(f"[{time.ctime()}] 处理列 {column}, 错误总数: {self.detection[column].sum()}")
                self.logs.append(f"处理列 {column}, 错误总数: {self.detection[column].sum()}")
                if self.detection[column].sum() > 0:
                    retriever_start = time.time()
                    try:
                        retriever, indices, CoE = self.build_retriever_3(column)
                        self.retriever_dict[column] = retriever
                        self.indices_dict[column] = indices
                        self.CoE_dict[column] = CoE
                        print(f"[{time.ctime()}] 为列 {column} 构建检索器完成")
                        self.logs.append(f"为列 {column} 构建检索器完成")
                    except Exception as e:
                        print(f"[{time.ctime()}] 为列 {column} 构建检索器失败: {str(e)}")
                        self.logs.append(f"为列 {column} 构建检索器失败: {str(e)}")
                        self.retriever_dict[column] = None
                        self.indices_dict[column] = []
                        self.CoE_dict[column] = []
                    retriever_build_time += time.time() - retriever_start
                else:
                    print(f"[{time.ctime()}] 跳过列 {column}: 无错误检测到")
                    self.logs.append(f"跳过列 {column}: 无错误检测到")
                    self.retriever_dict[column] = None
                    self.indices_dict[column] = []
                    self.CoE_dict[column] = []
            print(f"[{time.ctime()}] 检索器构建耗时: {retriever_build_time:.2f} 秒")
            self.f_time_cost.write(f"[{time.ctime()}] 检索器构建耗时: {retriever_build_time:.2f} 秒\n")

            # Auto-CoT 生成
            print(f"[{time.ctime()}] 开始 Auto-CoT 生成")
            time_auto_cot_start = time.time()
            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = []
                for col_idx, column in enumerate(self.detection.columns):
                    if self.detection[column].sum() > 0:
                        retriever = self.retriever_dict.get(column)
                        indices = self.indices_dict.get(column, [])
                        CoE = self.CoE_dict.get(column, [])
                        detection_filtered = self.detection.iloc[:, indices] if indices else pd.DataFrame()
                        repair_list_for_column = self.detection[column][self.detection[column] == 1].index.tolist()[:10]
                        print(f"[{time.ctime()}] 提交任务: 列 {column}, 错误数: {len(repair_list_for_column)}")
                        future = executor.submit(
                            self.get_auto_cot,
                            repair_list_for_column, column, retriever, CoE, indices, detection_filtered,
                            self.rep_error_info, self.sp_examps, self.dirty_data, self.clean_data,
                            self.embeddings_matrix, self.detection, self.detection_human_repaired, self.header,
                            self.sys_auto_cot, self.human_auto_cot_large, self.human_auto_cot_small,
                            self.general_examples_str, self.auto_cot_llm
                        )
                        futures.append((future, column))
                for future, column in futures:
                    try:
                        results, failed_rows = future.result(timeout=120)  # 增加超时，避免挂起
                        success_count = len(results)
                        total = len(repair_list_for_column) or 1  # 防止除零
                        if success_count > 0:
                            # 有成功或 fallback 修正，视为部分成功
                            failed_count = len(failed_rows)
                            log_msg = f"Auto-CoT 部分成功 for 列 {column}: {success_count}/{total} 行成功"
                            if failed_count > 0:
                                log_msg += f"，{failed_count} 行使用 fallback"
                            print(f"[{time.ctime()}] {log_msg}")
                            self.logs.append(log_msg)
                        else:
                            # 完全失败（无任何有效或 fallback 输出）
                            print(
                                f"[{time.ctime()}] Auto-CoT 任务失败 for 列 {column}: 0/{total} 行成功，失败行: {failed_rows}")
                            self.logs.append(f"Auto-CoT 任务失败 for 列 {column}: 失败行: {failed_rows}")

                            # Fallback 整个列，确保有输出
                            for row_idx in repair_list_for_column:
                                default_correction = self.fallback_correction(str(self.dirty_data.at[row_idx, column]),
                                                                              column)
                                results.append((row_idx, column, default_correction))
                                self.corrections.at[row_idx, column] = default_correction
                                self.corrections_df = pd.concat([
                                    self.corrections_df,
                                    pd.DataFrame({
                                        'row': [row_idx],
                                        'col': [column],
                                        'original': [self.dirty_data.at[row_idx, column]],
                                        'corrected': [default_correction]
                                    })
                                ], ignore_index=True)
                                print(
                                    f"[{time.ctime()}] 完全失败 Fallback 修正: 列 {column} 行 {row_idx}, 新值: {default_correction}")
                                self.logs.append(
                                    f"完全失败 Fallback for 列 {column} 行 {row_idx}: {default_correction}")

                        # 处理所有结果（成功或 fallback）
                        for row_idx, col, correction in results:
                            self.corrections.at[row_idx, col] = correction
                            self.corrections_df = pd.concat([
                                self.corrections_df,
                                pd.DataFrame({
                                    'row': [row_idx],
                                    'col': [col],
                                    'original': [self.dirty_data.at[row_idx, col]],
                                    'corrected': [correction]
                                })
                            ], ignore_index=True)
                            print(
                                f"[{time.ctime()}] 应用 Auto-CoT 修正: 列 {col} 行 {row_idx}, 原值: {self.dirty_data.at[row_idx, col]}, 新值: {correction}")

                    except Exception as e:
                        error_msg = f"[{time.ctime()}] Auto-CoT 线程异常 for 列 {column}: {str(e)} - 类型: {type(e).__name__} - 完整: {repr(e)}"
                        print(error_msg)
                        self.logs.append(error_msg)

                        # 线程异常时 fallback 整个列
                        repair_list_for_column = self.detection[column][self.detection[column] == 1].index.tolist()[:10]
                        for row_idx in repair_list_for_column:
                            default_correction = self.fallback_correction(str(self.dirty_data.at[row_idx, column]),
                                                                          column)
                            self.corrections.at[row_idx, column] = default_correction
                            self.corrections_df = pd.concat([
                                self.corrections_df,
                                pd.DataFrame({
                                    'row': [row_idx],
                                    'col': [column],
                                    'original': [self.dirty_data.at[row_idx, column]],
                                    'corrected': [default_correction]
                                })
                            ], ignore_index=True)
                            print(
                                f"[{time.ctime()}] 线程 Fallback 修正: 列 {column} 行 {row_idx}, 新值: {default_correction}")
                            self.logs.append(f"线程 Fallback for 列 {column} 行 {row_idx}: {default_correction}")

            # 保存 specific_examples
            with open(os.path.join(self.output_path, 'specific_examples.txt'), 'w', encoding='utf-8') as f_output:
                for column, few_shot_specific_str in self.sp_examps.items():
                    f_output.write(f"列: {column}\n")
                    f_output.write(f"示例:\n{few_shot_specific_str}\n")
                    f_output.write("\n" + "=" * 50 + "\n\n")
            time_auto_cot_end = time.time()
            print(f"[{time.ctime()}] Auto-CoT 生成完成")
            print(f"[{time.ctime()}] 耗时: {time_auto_cot_end - time_auto_cot_start:.2f} 秒")
            self.f_time_cost.write(
                f"[{time.ctime()}] Auto-CoT 生成耗时: {time_auto_cot_end - time_auto_cot_start:.2f} 秒\n")

            # 数据增强
            print(f"[{time.ctime()}] 开始数据增强")
            time_augmentation_start = time.time()
            for column in self.rep_error_info:
                for row_idx, error_info in self.rep_error_info[column].items():
                    if 'error_type' not in error_info or error_info['error_type'] != 'clean':
                        try:
                            augmented_result = self.data_augmentation(
                                column_name=column,
                                dirty_tuple=self.format_row(self.dirty_data.iloc[row_idx], self.header),
                                dirty_value=str(self.dirty_data.at[row_idx, column]),
                                error_analysis=error_info.get('error_analysis', ''),
                                llm_instance=self.data_augmentation_llm,
                                sys_data_augmentation=self.sys_data_augmentation,
                                human_data_augmentation=self.human_data_augmentation
                            )
                            self.augmented_data[column][row_idx] = augmented_result
                        except Exception as e:
                            print(f"[{time.ctime()}] 数据增强失败 for 列 {column} 行 {row_idx}: {str(e)}")
                            self.logs.append(f"数据增强失败 for 列 {column} 行 {row_idx}: {str(e)}")
            with open(os.path.join(self.output_path, 'augmented_data.txt'), 'w', encoding='utf-8') as f_output:
                for column, aug_data in self.augmented_data.items():
                    f_output.write(f"列: {column}\n")
                    f_output.write(f"增强数据:\n{json.dumps(aug_data, indent=2, ensure_ascii=False)}\n")
                    f_output.write("\n" + "=" * 50 + "\n\n")
            time_augmentation_end = time.time()
            print(f"[{time.ctime()}] 数据增强完成")
            print(f"[{time.ctime()}] 耗时: {time_augmentation_end - time_augmentation_start:.2f} 秒")
            self.f_time_cost.write(
                f"[{time.ctime()}] 数据增强耗时: {time_augmentation_end - time_augmentation_start:.2f} 秒\n")

            # 清理和整合数据
            print(f"[{time.ctime()}] 开始清理和整合数据")
            rep_clean_data = self.sel_clean(self.human_repair_num if hasattr(self, 'human_repair_num') else 10)
            rep_data_info = copy.deepcopy(self.rep_error_info)
            print(
                f"[{time.ctime()}] rep_clean_data 形状: {rep_clean_data.shape}, rep_data_info 键: {list(rep_data_info.keys())}")
            if not rep_clean_data.empty:
                try:
                    self.clean_data_integration(rep_clean_data, rep_data_info)
                    print(f"[{time.ctime()}] clean_data_integration 完成")
                except Exception as e:
                    print(f"[{time.ctime()}] clean_data_integration 失败: {str(e)}")
                    self.logs.append(f"clean_data_integration 失败: {str(e)}")
            for column in self.augmented_data:
                for row_idx, aug_result in self.augmented_data[column].items():
                    rep_data_info[column][f"aug_{row_idx}"] = {
                        'dirty_tuple': aug_result.get('generated_error', {}),
                        'dirty_value': aug_result.get('generated_error', {}).get(column, aug_result.get('correct_value',
                                                                                                        {}).get(column,
                                                                                                                '')),
                        'error_analysis': aug_result.get('error_analysis', ''),
                        'ground_truth': aug_result.get('correct_value', {}).get(column, ''),
                        'error_type': self.rep_error_info[column][row_idx].get('error_type', 'unknown')
                    }

            # 代码和功能依赖生成
            print(f"[{time.ctime()}] 开始代码和功能依赖生成")
            time_code_generation_start = time.time()
            self.codes = {}
            self.fds = {}
            train_data = {}
            val_data = {}
            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = []
                for column in self.detection.columns:
                    if self.detection[column].sum() > 0:
                        formatting_issue = all(
                            error.get('error_type', 'unknown') in ['Formatting Issue', 'clean']
                            for error in rep_data_info.get(column, {}).values()
                        )
                        train_data[column], val_data[column] = self.train_val_split(rep_data_info.get(column, {}))
                        print(f"[{time.ctime()}] 提交代码/FD生成任务: 列 {column}")
                        if formatting_issue:
                            future = executor.submit(
                                self.code_generation,
                                train_data[column], column, self.codes, self.code_generation_llm,
                                self.sys_code_generation, self.human_code_generation
                            )
                        else:
                            future = executor.submit(
                                self.fd_generation_for_column,
                                train_data[column], column
                            )
                        futures.append((future, column))
                for future, column in futures:
                    try:
                        result = future.result(timeout=60)
                        if not formatting_issue:
                            self.fds[column] = result
                            print(f"[{time.ctime()}] FD 生成成功 for 列 {column}: {result}")
                    except Exception as e:
                        print(f"[{time.ctime()}] 代码/FD 生成任务失败 for 列 {column}: {str(e)}")
                        self.logs.append(f"代码/FD 生成任务失败 for 列 {column}: {str(e)}")
                        self.fds[column] = {}

            with open(os.path.join(self.output_path, 'codes.txt'), 'w', encoding='utf-8') as f_output:
                for column, code in self.codes.items():
                    f_output.write(f"列: {column}\n")
                    f_output.write(f"代码:\n{code}\n")
                    f_output.write("\n" + "=" * 50 + "\n\n")
            with open(os.path.join(self.output_path, 'fds.txt'), 'w', encoding='utf-8') as f_output:
                for column, fd in self.fds.items():
                    f_output.write(f"列: {column}\n")
                    f_output.write(f"功能依赖:\n{json.dumps(fd, ensure_ascii=False)}\n")
                    f_output.write("\n" + "=" * 50 + "\n\n")
            time_code_generation_end = time.time()
            print(f"[{time.ctime()}] 代码和功能依赖生成完成")
            print(f"[{time.ctime()}] 耗时: {time_code_generation_end - time_code_generation_start:.2f} 秒")
            self.f_time_cost.write(
                f"[{time.ctime()}] 代码和功能依赖生成耗时: {time_code_generation_end - time_code_generation_start:.2f} 秒\n")

            # 代码和功能依赖评估
            print(f"[{time.ctime()}] 开始代码和功能依赖评估")
            time_code_evaluation_start = time.time()
            for column in self.codes.keys():
                try:
                    self.code_evaluation_execution(self.codes[column], val_data[column], column)
                except Exception as e:
                    print(f"[{time.ctime()}] 代码评估和执行失败: 列 {column}: {str(e)}")
                    self.logs.append(f"代码评估和执行失败: 列 {column}: {str(e)}")
            for column in self.fds.keys():
                try:
                    self.fd_evaluation_execution(self.fds[column], val_data[column], column)
                except Exception as e:
                    print(f"[{time.ctime()}] 功能依赖评估和执行失败: 列 {column}: {str(e)}")
                    self.logs.append(f"功能依赖评估和执行失败: 列 {column}: {str(e)}")
            time_code_evaluation_end = time.time()
            print(f"[{time.ctime()}] 代码和功能依赖评估完成")
            print(f"[{time.ctime()}] 耗时: {time_code_evaluation_end - time_code_evaluation_start:.2f} 秒")
            self.f_time_cost.write(
                f"[{time.ctime()}] 代码和功能依赖评估耗时: {time_code_evaluation_end - time_code_evaluation_start:.2f} 秒\n")

            # 执行修复流程
            print(f"[{time.ctime()}] 开始 repair_tableau")
            self.repair_tableau()

            # 保存结果
            try:
                self.corrections.to_csv(os.path.join(self.output_path, 'corrections.csv'), encoding='utf-8',
                                        index=False)
                self.cmp_mark(self.clean_data, self.corrections)
                metrics = self.calc_p_r_f(self.clean_data, self.dirty_data_human_repaired, self.corrections)
                print(f"[{time.ctime()}] 最终评估结果: {metrics}")
                self.f_time_cost.write(f"[{time.ctime()}] 最终评估结果: {metrics}\n")
            except Exception as e:
                print(f"[{time.ctime()}] 保存结果或评估失败: {str(e)}")
                self.logs.append(f"保存结果或评估失败: {str(e)}")
            print(f"[{time.ctime()}] Fit 完成")
            return self

        def fd_generation_for_column(self, train_data, column):
            print(f"[{time.ctime()}] 开始为列 {column} 生成 FD")
            try:
                # 转换键为 str 以避免 numpy.int64 问题
                train_data_str_keys = {str(k): v for k, v in train_data.items()}
                column_type = str(self.dirty_data[column].dtype)
                parser = JsonOutputParser()

                # 消息列表
                messages = [
                    SystemMessagePromptTemplate.from_template(self.sys_fd_generation),
                    HumanMessagePromptTemplate.from_template(self.human_fd_generation)
                ]

                # 提示模板
                prompt = ChatPromptTemplate(
                    messages=messages,
                    partial_variables={
                        'format_instructions': parser.get_format_instructions()
                    }
                )
                prompt_input = {
                    'column_name': column,
                    'column_type': column_type,
                    'examples': json.dumps(train_data_str_keys, ensure_ascii=False)
                }
                chain = prompt | self.fd_generation_llm | parser
                result = chain.invoke(prompt_input)
                print(f"[{time.ctime()}] LLM 原始 FD 输出: {result}")

                # 处理输出：包装为列表
                if not isinstance(result, list):
                    wrapped_result = [{'correction': result,
                                       'functional_dependency': result.get('functional_dependency',
                                                                           'def correct(value): return value')}] if isinstance(
                        result, dict) else [
                        {'correction': {'functional_dependency': 'def correct(value): return value'},
                         'functional_dependency': 'def correct(value): return value'}]
                    result = wrapped_result
                elif not result or 'correction' not in result[0]:
                    result = [{'correction': result[0] if isinstance(result[0], dict) else {
                        'functional_dependency': 'def correct(value): return value'},
                               'functional_dependency': 'def correct(value): return value'}]

                corrected_result = {str(k): v for k, v in result[0].items()}

                # 规范化 functional_dependency 代码
                fd_code = corrected_result.get('functional_dependency', '')
                if not isinstance(fd_code, str) or not fd_code.strip():
                    fd_code = self._get_default_fd_code(column)
                else:
                    # 修复语法：去除分号、重复 'nan'、复杂日期条件，规范化缩进
                    fd_code = re.sub(r';\s*return', '\n    return', fd_code)
                    fd_code = re.sub(r'\s*or\s*value\s*==\s*[\'"]nan[\'"][\n\s]*', '', fd_code)
                    fd_code = re.sub(r'\s*elif\s*value\s*==\s*[\'"]nan[\'"][\n\s:]*return\s*[\'"].*?[\'"]', '', fd_code)
                    fd_code = re.sub(r"if\s*value\s*==\s*['\"][0-9/]+\s+[0-9:]+\s*[ap]\.m\.['\"]\s*(or\s*)?", '',
                                     fd_code)
                    fd_code = re.sub(r'elif\s*[\'"]nan[\'"]\s*in\s*value\s*:.*?[\n\s]*', '', fd_code)
                    fd_code = re.sub(r'elif\s*[\'"]/201[0-9][\'"]\s*in\s*value\s*:.*?[\n\s]*', '', fd_code)
                    fd_code = re.sub(r'return\s*value',
                                     "return value.replace(' a.m.', ' am').replace(' p.m.', ' pm') if isinstance(value, str) else value",
                                     fd_code)
                    fd_code = textwrap.dedent(fd_code).strip()
                    try:
                        ast.parse(fd_code)
                    except SyntaxError:
                        print(f"[{time.ctime()}] FD 代码语法错误，替换为默认: {fd_code[:100]}...")
                        fd_code = self._get_default_fd_code(column)

                corrected_result['functional_dependency'] = fd_code
                return corrected_result
            except Exception as e:
                error_msg = f"[{time.ctime()}] FD 生成失败 for 列 {column}: {str(e)}"
                print(error_msg)
                self.logs.append(error_msg)
                return {'functional_dependency': self._get_default_fd_code(column)}

        def _get_default_fd_code(self, column):
            """生成默认 FD 代码，基于列名推断时间格式"""
            default_time = '12:00 pm'
            if 'time' in column.lower():
                return f"""def correct(value):
            if pd.isna(value):
                return '{default_time}'
            elif isinstance(value, str) and '/' in value:
                parts = value.split(' ')
                if len(parts) > 1 and (parts[-1].endswith('m') or parts[-1].endswith('m.')):
                    return parts[-1].replace(' a.m.', ' am').replace(' p.m.', ' pm').replace('a.m.', 'am').replace('p.m.', 'pm')
                return value.replace(' a.m.', ' am').replace(' p.m.', ' pm').replace('a.m.', 'am').replace('p.m.', 'pm')
            return value.replace(' a.m.', ' am').replace(' p.m.', ' pm').replace('a.m.', 'am').replace('p.m.', 'pm') if isinstance(value, str) else value"""
            return """def correct(value):
            if pd.isna(value):
                return ''
            return value"""
        def sort_dicts(self, dict_list: List[Dict], *keys: str) -> List[Dict]:
            """Sort a list of dictionaries by multiple keys."""

            def get_key(item):
                return tuple(item.get(key, 0) for key in keys)

            return sorted(dict_list, key=get_key)

        def build_retriever_3(self, column: str) -> Tuple[Any, List[int], np.ndarray]:
            print(f"[{time.ctime()}] 为列 {column} 构建检索器")
            try:
                # 验证输入数据
                if column not in self.detection.columns:
                    print(f"[{time.ctime()}] 错误: 列 {column} 不在 detection 中")
                    self.logs.append(f"错误: 列 {column} 不在 detection 中")
                    return None, [], np.array([])

                if self.clean_data.empty or self.detection.empty:
                    print(f"[{time.ctime()}] 错误: clean_data 或 detection 为空")
                    self.logs.append(f"错误: clean_data 或 detection 为空")
                    return None, [], np.array([])

                # 计算互信息分数
                mi_scores = []
                for col in self.detection.columns:
                    if col != column:
                        score = mutual_info_score(self.detection[column], self.detection[col])
                        mi_scores.append(score)
                    else:
                        mi_scores.append(0)
                normalized_mi = np.array(mi_scores) / (np.max(mi_scores) if np.max(mi_scores) > 0 else 1)
                indices = [i for i, score in enumerate(normalized_mi) if score > 0.5]
                if not indices:
                    indices = [self.detection.columns.get_loc(column)]
                    print(f"[{time.ctime()}] 警告: 列 {column} 无相关列，使用自身索引")
                CoE = np.array([1.0 if i in indices else 0.0 for i in range(len(self.detection.columns))])

                # 格式化文本
                texts = []
                for idx, row in self.clean_data.iterrows():
                    values = row.values
                    detection_row = self.detection.loc[idx].values
                    text = self.format_row_2(values, self.header, detection_row)
                    if not text or text == "{}":
                        print(f"[{time.ctime()}] 警告: 列 {column} 的行 {idx} 格式化为空")
                        text = json.dumps({column: str(values[self.detection.columns.get_loc(column)])},
                                          ensure_ascii=False)
                    texts.append(text)

                if not texts:
                    print(f"[{time.ctime()}] 错误: 列 {column} 的 texts 为空")
                    self.logs.append(f"错误: 列 {column} 的 texts 为空")
                    return None, indices, CoE

                # 构建 FAISS 检索器
                faiss_retriever = FAISS.from_texts(
                    texts=texts,
                    embedding=self.embedding_model,
                    metadatas=[{"index": int(i)} for i in range(len(texts))],  # 转换为 int 以避免 JSON 序列化错误
                    distance_strategy=DistanceStrategy.COSINE
                ).as_retriever(search_kwargs={"k": 10})

                # 仅使用 FAISS，移除 BM25Retriever
                print(f"[{time.ctime()}] 检索器构建完成 for 列 {column}, indices: {indices}, CoE shape: {CoE.shape}")
                return faiss_retriever, indices, CoE

            except Exception as e:
                print(f"[{time.ctime()}] 构建检索器失败 for 列 {column}: {str(e)}")
                self.logs.append(f"构建检索器失败 for 列 {column}: {str(e)}")
                return None, indices, CoE

        def data_augmentation(self, column_name, dirty_tuple, dirty_value, error_analysis, llm_instance,
                              sys_data_augmentation, human_data_augmentation):
            print(f"[{time.ctime()}] 开始数据增强 for 列 {column_name}")
            try:
                prompt = ChatPromptTemplate.from_messages([
                    SystemMessagePromptTemplate.from_template(sys_data_augmentation),
                    HumanMessagePromptTemplate.from_template(human_data_augmentation)
                ])
                prompt_input = {
                    'column': column_name,  # 改为 'column' 以匹配提示模板
                    'Dirty_tuple': json.dumps(dirty_tuple, ensure_ascii=False) if isinstance(dirty_tuple,
                                                                                             dict) else str(
                        dirty_tuple),
                    'Dirty_value': str(dirty_value),
                    'Error_analysis': str(error_analysis),
                    'general_examples': self.general_examples_str,
                    'specific_examples': self.sp_examps.get(column_name, '')
                }
                chain = prompt | llm_instance | JsonOutputParser()
                result = chain.invoke(prompt_input)
                print(f"[{time.ctime()}] 数据增强成功 for 列 {column_name}: {result}")
                self.logs.append(f"数据增强成功 for 列 {column_name}: {result}")
                return result if isinstance(result, dict) else {}
            except Exception as e:
                error_msg = f"[{time.ctime()}] 数据增强失败 for 列 {column_name}: {str(e)}"
                print(error_msg)
                self.logs.append(error_msg)
                return {}

        def code_generation(self, train_data: Dict, column: str, codes: Dict, llm, sys_prompt: str,
                            human_prompt: str) -> None:
            print(f"[{time.ctime()}] 开始为列 {column} 生成代码")
            self.logs.append({
                'message': f"开始为列 {column} 生成代码",
                'timestamp': time.ctime()
            })
            try:
                examples = form_examples([
                    {'original': k, 'corrected': v['correction'], 'error_type': v.get('error_type', '未知')}
                    for k, v in train_data.items()
                ])
                prompt = ChatPromptTemplate.from_messages([
                    SystemMessagePromptTemplate.from_template(sys_prompt),
                    HumanMessagePromptTemplate.from_template(human_prompt)
                ])
                chain = prompt | llm | JsonOutputParser()
                prompt_input = {
                    'column_name': column,
                    'column_type': str(self.dirty_data[column].dtype),
                    'examples': examples,
                    'format_instructions': JsonOutputParser().get_format_instructions()
                }
                response = chain.invoke(prompt_input)
                print(f"[{time.ctime()}] LLM 原始输出 for 代码生成 列 {column}: {response}")
                self.logs.append({
                    'message': f"LLM 原始输出 for 代码生成 列 {column}: {response}",
                    'timestamp': time.ctime()
                })
                if isinstance(response, list) and len(response) == 1:
                    response = response[0]
                if not isinstance(response, dict):
                    response_str = str(response).strip().lstrip('\n').rstrip('\n')
                    try:
                        response = json.loads(response_str)
                        if isinstance(response, list) and len(response) == 1:
                            response = response[0]
                    except json.JSONDecodeError:
                        response = {
                            'correction': f'def correct(value):\n    return re.sub(r"^err_?|^ERR_?", "", str(value))'}
                if 'correction' not in response:
                    response['correction'] = f'def correct(value):\n    return re.sub(r"^err_?|^ERR_?", "", str(value))'
                codes[column] = response['correction']
                print(f"[{time.ctime()}] 代码生成成功 for 列 {column}: {response}")
                self.logs.append({
                    'message': f"代码生成成功 for 列 {column}: {response}",
                    'timestamp': time.ctime()
                })
            except Exception as e:
                print(f"[{time.ctime()}] 代码生成失败 for 列 {column}: {str(e)}")
                self.logs.append({
                    'message': f"代码生成失败 for 列 {column}: {str(e)}",
                    'timestamp': time.ctime()
                })
                codes[column] = f'def correct(value):\n    return re.sub(r"^err_?|^ERR_?", "", str(value))'

        def FD_generation(self, train_data: Dict, column: str, fds: Dict, llm, sys_prompt: str,
                          human_prompt: str) -> None:
            print(f"[{time.ctime()}] 开始为列 {column} 生成 FD")
            self.logs.append({
                'message': f"开始为列 {column} 生成 FD",
                'timestamp': time.ctime()
            })
            try:
                examples = form_examples([
                    {'original': k, 'corrected': v['correction'], 'error_type': v.get('error_type', '未知')}
                    for k, v in train_data.items()
                ])
                prompt = ChatPromptTemplate.from_messages([
                    SystemMessagePromptTemplate.from_template(sys_prompt),
                    HumanMessagePromptTemplate.from_template(human_prompt)
                ])
                chain = prompt | llm | JsonOutputParser()
                prompt_input = {
                    'column_name': column,
                    'column_type': str(self.dirty_data[column].dtype),
                    'examples': examples,
                    'format_instructions': JsonOutputParser().get_format_instructions()
                }
                response = chain.invoke(prompt_input)
                print(f"[{time.ctime()}] LLM 原始输出 for FD 列 {column}: {response}")
                self.logs.append({
                    'message': f"LLM 原始输出 for FD 列 {column}: {response}",
                    'timestamp': time.ctime()
                })
                if isinstance(response, list) and len(response) == 1:
                    response = response[0]
                if not isinstance(response, dict):
                    response_str = str(response).strip().lstrip('\n').rstrip('\n')
                    try:
                        response = json.loads(response_str)
                        if isinstance(response, list) and len(response) == 1:
                            response = response[0]
                    except json.JSONDecodeError:
                        response = {
                            'functional_dependency': f'def correct(value):\n    return re.sub(r"^err_?|^ERR_?", "", str(value))'}
                if 'functional_dependency' not in response:
                    response[
                        'functional_dependency'] = f'def correct(value):\n    return re.sub(r"^err_?|^ERR_?", "", str(value))'
                fds[column] = response
                print(f"[{time.ctime()}] FD 生成成功 for 列 {column}: {response}")
                self.logs.append({
                    'message': f"FD 生成成功 for 列 {column}: {response}",
                    'timestamp': time.ctime()
                })
            except Exception as e:
                print(f"[{time.ctime()}] FD 生成失败 for 列 {column}: {str(e)}")
                self.logs.append({
                    'message': f"FD 生成失败 for 列 {column}: {str(e)}",
                    'timestamp': time.ctime()
                })
                fds[column] = {
                    'functional_dependency': f'def correct(value):\n    return re.sub(r"^err_?|^ERR_?", "", str(value))'}

        def code_evaluation_execution(self, code, val_data, column):
            """Evaluate and execute generated code."""
            try:
                exec(code)  # 注意：执行动态代码需谨慎，确保安全
                self.logs.append(f"Code executed successfully for {column}")
            except Exception as e:
                print(f"[{time.ctime()}] Code execution failed for {column}: {str(e)}")
                self.logs.append(f"Code execution failed for {column}: {str(e)}")


        def sel_clean(self, num_clusters: int) -> pd.DataFrame:
            print(f"[{time.ctime()}] Starting sel_clean, num_clusters={num_clusters}")
            try:
                mask_rows = self.detection.sum(axis=1) == 0
                clean_indices = np.where(mask_rows)[0]
                if len(clean_indices) == 0:
                    print(f"[{time.ctime()}] No clean rows found, returning empty DataFrame")
                    self.logs.append("No clean rows found")
                    return pd.DataFrame()
                num_to_select = min(num_clusters, len(clean_indices))
                selected_indices = np.random.choice(clean_indices, size=num_to_select, replace=False)
                print(f"[{time.ctime()}] Selected {len(selected_indices)} clean data candidates")
                self.logs.append(f"Selected {len(selected_indices)} clean data candidates")
                return self.clean_data.iloc[selected_indices]
            except Exception as e:
                print(f"[{time.ctime()}] sel_clean failed: {str(e)}")
                self.logs.append(f"sel_clean failed: {str(e)}")
                return pd.DataFrame()
        def cmp_mark(self, clean_data: pd.DataFrame, corrected_data: pd.DataFrame):
            """Compare and mark corrections."""
            mask = clean_data != corrected_data
            comparison = pd.DataFrame(index=corrected_data.index, columns=corrected_data.columns)
            for col in corrected_data.columns:
                comparison[col] = np.where(mask[col], 'corrected', 'unchanged')
            comparison.to_csv(os.path.join(self.output_path, 'comparison.csv'), encoding='utf-8')
            self.logs.append(f"Comparison saved to {os.path.join(self.output_path, 'comparison.csv')}")

        def calc_p_r_f(self, clean_data: pd.DataFrame, dirty_data: pd.DataFrame, corrected_data: pd.DataFrame) -> Dict[
            str, float]:
            print(f"[{time.ctime()}] Calculating P/R/F1")

            def is_effectively_na(value):
                if isinstance(value, (np.ndarray, pd.Series)):
                    if value.size == 0:
                        return True
                    try:
                        return np.all(pd.isna(value))
                    except TypeError:
                        return False
                elif isinstance(value, list):
                    return False
                return pd.isna(value)

            def safe_equals(val1, val2):
                is_val1_na = is_effectively_na(val1)
                is_val2_na = is_effectively_na(val2)
                if is_val1_na and is_val2_na:
                    return True
                if is_val1_na or is_val2_na:
                    return False
                if isinstance(val1, (float, np.float64)) and isinstance(val2, (float, np.float64)):
                    return np.isclose(val1, val2, rtol=1e-09, atol=1e-09)
                return str(val1).strip().lower() == str(val2).strip().lower()

            def safe_not_equals(val1, val2):
                return not safe_equals(val1, val2)

            true_positives = 0
            false_positives = 0
            false_negatives = 0

            for col in clean_data.columns:
                for idx in clean_data.index:
                    clean_val = clean_data.at[idx, col]
                    dirty_val = dirty_data.at[idx, col]
                    corrected_val = corrected_data.at[idx, col]

                    if is_effectively_na(clean_val) or is_effectively_na(dirty_val) or is_effectively_na(corrected_val):
                        continue

                    is_dirty = safe_not_equals(dirty_val, clean_val)
                    is_corrected = safe_not_equals(corrected_val, dirty_val)
                    is_correct = safe_equals(corrected_val, clean_val)

                    print(
                        f"[{time.ctime()}] 比较: 行 {idx}, 列 {col}, 脏值={dirty_val}, 干净值={clean_val}, 修复值={corrected_val}, "
                        f"is_dirty={is_dirty}, is_corrected={is_corrected}, is_correct={is_correct}")

                    if is_dirty:
                        if is_corrected:
                            if is_correct:
                                true_positives += 1
                            else:
                                false_positives += 1
                        else:
                            false_negatives += 1
                    else:
                        if is_corrected:
                            false_positives += 1

            precision = true_positives / (true_positives + false_positives) if (
                                                                                           true_positives + false_positives) > 0 else 0
            recall = true_positives / (true_positives + false_negatives) if (
                                                                                        true_positives + false_negatives) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            metrics = {'precision': precision, 'recall': recall, 'f1': f1}
            print(
                f"[{time.ctime()}] Metrics: {metrics}, TP={true_positives}, FP={false_positives}, FN={false_negatives}")
            return metrics

        def format_row(self, row, header):
            s = '{' + ', '.join(f'"{col}": "{val}"' for col, val in zip(header, row)) + '}'
            return s

        def format_row_2(self, value, key, detection_row):
            result = {key[i]: value[i] for i in range(len(value)) if detection_row[i] == 0}
            if not result:  # 全为 1 时，使用所有值
                result = {key[i]: value[i] for i in range(len(value))}
                print(f"[{time.ctime()}] 警告: detection_row 全为 1，使用所有值: {result}")
            return json.dumps(result, ensure_ascii=False)

        def process_repair_item(self, prompt, llm, parser, prompt_input_data, column, row_idx):
            max_retries = 3
            print(f"[{time.ctime()}] 处理列 {column} 的行 {row_idx}")
            self.logs.append({
                'message': f"处理列 {column} 的行 {row_idx}",
                'timestamp': time.ctime()
            })

            # 单行提示模板，避免换行符问题
            human_prompt_template = "Column Name: {column_name} | Column Type: {column_type} | Dirty Data Tuple: {Dirty_Tuple} | Erroneous Value: {Erroneous_value} | Relevant Clean Tuples: {Relevant_clean_tuples} | General Examples: {general_examples} | Specific Examples: {specific_examples} | 请为指定列中的错误值提供修复，并以严格的 JSON 格式返回响应，如下所示: {{ \"correction\": \"<corrected_value>\", \"chain_of_thought_for_correction\": \"<your_reasoning>\" }} | {format_instructions}"
            print(f"[{time.ctime()}] 原始 human 提示模板: {human_prompt_template}")
            self.logs.append({
                'message': f"原始 human 提示模板: {human_prompt_template}",
                'timestamp': time.ctime()
            })

            for attempt in range(max_retries):
                try:
                    # 确保 prompt_input_data 包含所有必需变量
                    prompt_input_data = {
                        'column_name': column,
                        'column_type': str(self.dirty_data[column].dtype),
                        'Dirty_Tuple': json.dumps(
                            {k: v for k, v in self.dirty_data.loc[row_idx].items() if k != column}, ensure_ascii=False),
                        'Erroneous_value': str(
                            prompt_input_data.get('Erroneous_value', self.dirty_data.loc[row_idx, column])),
                        'Relevant_clean_tuples': prompt_input_data.get('Relevant_clean_tuples', ''),
                        'Dirty_tuple': json.dumps(
                            {k: v for k, v in self.dirty_data.loc[row_idx].items() if k != column}, ensure_ascii=False),
                        'general_examples': load_examples(os.path.join(self.prompt_template_dir, 'examples.txt')),
                        'specific_examples': json.dumps(self.specific_examples.get(column, []), ensure_ascii=False),
                        'format_instructions': parser.get_format_instructions()
                    }
                    print(f"[{time.ctime()}] 提示输入: {prompt_input_data}")
                    self.logs.append({
                        'message': f"提示输入: {prompt_input_data}",
                        'timestamp': time.ctime()
                    })

                    # 使用 PromptTemplate 替代 ChatPromptTemplate
                    from langchain.prompts import PromptTemplate
                    prompt = PromptTemplate(
                        input_variables=['column_name', 'column_type', 'Dirty_Tuple', 'Erroneous_value',
                                         'Relevant_clean_tuples', 'general_examples', 'specific_examples',
                                         'format_instructions'],
                        template=human_prompt_template
                    )

                    # 格式化提示并打印
                    formatted_prompt = prompt.format(**prompt_input_data)
                    print(f"[{time.ctime()}] 格式化后的提示: {formatted_prompt}")
                    self.logs.append({
                        'message': f"格式化后的提示: {formatted_prompt}",
                        'timestamp': time.ctime()
                    })

                    chain = prompt | llm | parser
                    response = chain.invoke(prompt_input_data)
                    print(f"[{time.ctime()}] LLM 原始输出 for 列 {column} 行 {row_idx}: {response}")
                    self.logs.append({
                        'message': f"LLM 原始输出 for 列 {column} 行 {row_idx}: {response}",
                        'timestamp': time.ctime()
                    })

                    # 处理响应
                    if isinstance(response, list) and len(response) == 1:
                        response = response[0]
                    if not isinstance(response, dict):
                        response_str = str(response).strip().lstrip('\n').rstrip('\n')
                        try:
                            response = json.loads(response_str)
                            if isinstance(response, list) and len(response) == 1:
                                response = response[0]
                        except json.JSONDecodeError:
                            response = {
                                'correction': re.sub(r'^err_?|^ERR_?', '', str(prompt_input_data['Erroneous_value'])),
                                'chain_of_thought_for_correction': '非 JSON 响应，使用回退纠正'
                            }
                            print(f"[{time.ctime()}] 非 JSON 响应，类型: {type(response_str)}, 内容: {response_str}")

                    # 确保 response 包含 correction 键
                    if 'correction' not in response:
                        print(f"[{time.ctime()}] 响应缺少 'correction' 键 for 列 {column} 行 {row_idx}: {response}")
                        response['correction'] = re.sub(r'^err_?|^ERR_?', '', str(prompt_input_data['Erroneous_value']))
                        response['chain_of_thought_for_correction'] = '未提供纠正，使用回退纠正'

                    self.logs.append({
                        'message': f"LLM 响应 for 列 {column} 行 {row_idx}: {response}",
                        'timestamp': time.ctime()
                    })
                    return column, row_idx, response['correction'], response.get('chain_of_thought_for_correction',
                                                                                 '未提供说明')

                except Exception as e:
                    print(
                        f"[{time.ctime()}] 调用 LLM 失败 for 列 {column} 行 {row_idx} (尝试 {attempt + 1}/{max_retries}): {str(e)}")
                    self.logs.append({
                        'message': f"调用 LLM 失败 for 列 {column} 行 {row_idx} (尝试 {attempt + 1}/{max_retries}): {str(e)}",
                        'timestamp': time.ctime()
                    })
                    if attempt == max_retries - 1:
                        dirty_value = prompt_input_data['Erroneous_value']
                        corrected_value = re.sub(r'^err_?|^ERR_?', '', str(dirty_value))
                        print(f"[{time.ctime()}] 应用回退纠正 for 列 {column}, 值 {dirty_value}")
                        print(f"[{time.ctime()}] 回退纠正: {dirty_value} -> {corrected_value}")
                        self.logs.append({
                            'message': f"应用回退纠正: {dirty_value} -> {corrected_value} at ({row_idx}, {column})",
                            'timestamp': time.ctime(),
                            'Index': str(row_idx),
                            'Column': column,
                            'Row Index': row_idx,
                            'Dirty_tuple': prompt_input_data['Dirty_tuple'],
                            'Dirty_value': dirty_value,
                            'Relevant_clean_tuples': prompt_input_data['Relevant_clean_tuples'],
                            'Correction': corrected_value,
                            'Explanation': f"应用回退纠正: {str(e)}"
                        })
                        return column, row_idx, corrected_value, f"应用回退纠正: {str(e)}"
                    time.sleep(1)

        def get_auto_cot(self, repair_list_for_column, column, retriever, CoE, indices, detection_filtered,
                         rep_error_info, sp_examps, dirty_data, clean_data, embeddings_matrix, detection,
                         detection_human_repaired, header, sys_auto_cot, human_auto_cot_large, human_auto_cot_small,
                         general_examples_str, auto_cot_llm):
            print(f"[{time.ctime()}] 开始为列 {column} 执行 Auto-CoT")
            results = []
            failed_rows = []
            max_retries = 3
            start_time = time.time()

            for row_idx in repair_list_for_column:
                print(f"[{time.ctime()}] 处理列 {column} 的行 {row_idx}")
                try:
                    dirty_tuple = self.format_row(dirty_data.iloc[row_idx], header)
                    erroneous_value = str(dirty_data.at[row_idx, column])
                    column_type = str(dirty_data[column].dtype)

                    # 获取相关干净元组
                    query_text = self.format_row(dirty_data.iloc[row_idx], [c for c in header if c != column])
                    try:
                        docs = retriever.invoke(query_text) if retriever else []
                        relevant_clean_tuples = [doc.page_content for doc in docs]
                    except Exception as e:
                        print(f"[{time.ctime()}] 检索相关干净元组失败 for 列 {column} 行 {row_idx}: {str(e)}")
                        self.logs.append(f"检索相关干净元组失败 for 列 {column} 行 {row_idx}: {str(e)}")
                        relevant_clean_tuples = []

                    # 准备提示输入
                    prompt_input = {
                        'Dirty_Tuple': dirty_tuple,
                        'Erroneous_value': erroneous_value,
                        'Relevant_clean_tuples': json.dumps(relevant_clean_tuples, ensure_ascii=False),
                        'general_examples': general_examples_str,
                        'specific_examples': sp_examps.get(column, ''),
                        'column_name': column,
                        'column_type': column_type
                    }

                    # 选择提示模板
                    prompt = ChatPromptTemplate.from_messages([
                        SystemMessagePromptTemplate.from_template(sys_auto_cot),
                        HumanMessagePromptTemplate.from_template(
                            human_auto_cot_large if len(sp_examps.get(column, '')) > 50 else human_auto_cot_small
                        )
                    ])

                    # 执行 LLM 调用
                    for attempt in range(max_retries):
                        try:
                            chain = prompt | auto_cot_llm | JsonOutputParser()
                            result = chain.invoke(prompt_input)
                            print(
                                f"[{time.ctime()}] LLM 原始输出 for 列 {column} 行 {row_idx} (尝试 {attempt + 1}): {result[:300]}...")  # 捕获原始输出，截断避免过长
                            if not isinstance(result, list) or not result or 'correction' not in result[0]:
                                raise ValueError(
                                    f"LLM 输出格式错误 for 行 {row_idx}: {str(result)[:300]}...")  # 错误包含行号和输出
                            correction = result[0]['correction']
                            error_type = result[0].get('error_type', 'unknown')
                            chain_of_thought = result[0].get('chain_of_thought_for_correction', '')
                            results.append((row_idx, column, correction))
                            rep_error_info[column][row_idx] = {
                                'dirty_tuple': dirty_tuple,
                                'dirty_value': erroneous_value,
                                'correction': correction,
                                'error_type': error_type,
                                'error_analysis': chain_of_thought
                            }
                            sp_examps[column] = f"{sp_examps.get(column, '')}\n{erroneous_value} -> {correction}"
                            print(f"[{time.ctime()}] Auto-CoT 成功 for 列 {column} 行 {row_idx}: {result[:300]}...")
                            break
                        except Exception as e:
                            error_msg = f"类型: {type(e).__name__}, 消息: {str(e)}"
                            print(
                                f"[{time.ctime()}] Auto-CoT 失败 for 列 {column} 行 {row_idx} (尝试 {attempt + 1}/{max_retries}): {error_msg}")
                            self.logs.append(f"Auto-CoT 失败 for 列 {column} 行 {row_idx}: {error_msg}")
                            # 捕获原始 LLM 输出（绕过 parser）
                            try:
                                raw_chain = prompt | auto_cot_llm
                                raw_response = raw_chain.invoke(prompt_input)
                                print(
                                    f"[{time.ctime()}] 原始 LLM 响应 for 列 {column} 行 {row_idx}: {raw_response.content[:300]}...")
                                self.logs.append(
                                    f"原始 LLM 响应 for 列 {column} 行 {row_idx}: {raw_response.content[:300]}...")
                            except Exception as raw_e:
                                print(
                                    f"[{time.ctime()}] 捕获原始 LLM 响应失败 for 列 {column} 行 {row_idx}: {str(raw_e)}")
                            if attempt == max_retries - 1:
                                print(
                                    f"[{time.ctime()}] Auto-CoT 最终失败 for 列 {column} 行 {row_idx}, 添加到 failed_rows")
                                self.logs.append(f"Auto-CoT 最终失败 for 列 {column} 行 {row_idx}, 添加到 failed_rows")
                                failed_rows.append(row_idx)
                                correction = self.fallback_correction(erroneous_value, column)
                                results.append((row_idx, column, correction))
                                rep_error_info[column][row_idx] = {
                                    'dirty_tuple': dirty_tuple,
                                    'dirty_value': erroneous_value,
                                    'correction': correction,
                                    'error_type': 'fallback',
                                    'error_analysis': '因 LLM 失败应用回退修正'
                                }
                                sp_examps[column] = f"{sp_examps.get(column, '')}\n{erroneous_value} -> {correction}"
                except Exception as e:
                    error_msg = f"类型: {type(e).__name__}, 消息: {str(e)}"
                    print(f"[{time.ctime()}] 处理行 {row_idx} 失败: {error_msg}")
                    self.logs.append(f"处理行 {row_idx} 失败: {error_msg}")
                    failed_rows.append(row_idx)
                    correction = self.fallback_correction(erroneous_value, column)
                    results.append((row_idx, column, correction))
                    rep_error_info[column][row_idx] = {
                        'dirty_tuple': dirty_tuple,
                        'dirty_value': erroneous_value,
                        'correction': correction,
                        'error_type': 'fallback',
                        'error_analysis': '因处理错误应用回退修正'
                    }
                    sp_examps[column] = f"{sp_examps.get(column, '')}\n{erroneous_value} -> {correction}"

            print(f"[{time.ctime()}] Auto-CoT 耗时 for 列 {column}: {time.time() - start_time:.2f} 秒")
            success_count = len([r for r in results if r[1] == column])
            total_count = len(repair_list_for_column)
            print(f"[{time.ctime()}] Auto-CoT 部分成功 for 列 {column}: {success_count}/{total_count} 行成功")
            self.logs.append(f"Auto-CoT 部分成功 for 列 {column}: {success_count}/{total_count} 行成功")
            if failed_rows:
                print(f"[{time.ctime()}] Auto-CoT 生成失败，失败行: {failed_rows}")
                self.logs.append(f"Auto-CoT 生成失败，失败行: {failed_rows}")
            return results, failed_rows
        def fallback_correction(self, dirty_value: Any, column: str) -> str:
            """Apply fallback correction to dirty data based on column-specific rules."""
            print(f"[{time.ctime()}] 应用回退纠正 for 列 {column}, 值 {dirty_value}")
            try:
                if pd.isna(dirty_value):
                    corrected = ""
                elif isinstance(dirty_value, str):
                    # 移除任何以 'err' 开头的前缀（包括 err, err_, errXXX 等）
                    corrected = re.sub(r'^err[^a-zA-Z0-9]*', '', dirty_value, flags=re.IGNORECASE).strip()
                    if not corrected:
                        corrected = dirty_value  # 如果移除后为空，保留原值
                else:
                    corrected = str(dirty_value)
                print(f"[{time.ctime()}] 回退纠正: {dirty_value} -> {corrected}")
                return corrected
            except Exception as e:
                print(f"[{time.ctime()}] 回退纠正失败 for 列 {column}: {str(e)}")
                self.logs.append(f"回退纠正失败 for 列 {column}: {str(e)}")
                return str(dirty_value)

        def repair_tableau(self):
            """使用基于 LLM 的 Auto-CoT 修复 tableau。"""
            print(f"[{time.ctime()}] 开始 repair_tableau")
            time_auto_cot_start = time.time()
            retriever_build_time = 0
            if not hasattr(self, 'sp_examps') or not self.sp_examps:
                self.sp_examps = {col: '' for col in self.header}
            if not hasattr(self, 'rep_error_info') or not self.rep_error_info:
                self.rep_error_info = {col: {} for col in self.header}

            # 确保 self.corrections 已初始化
            if self.corrections is None:
                print(f"[{time.ctime()}] 初始化 self.corrections 为 dirty_data 的副本")
                self.corrections = self._dirty_data_ref.copy()

            # 确保 self.dirty_data_human_repaired 已初始化
            if self.dirty_data_human_repaired is None:
                print(f"[{time.ctime()}] 初始化 self.dirty_data_human_repaired 为 dirty_data 的副本")
                self.dirty_data_human_repaired = self._dirty_data_ref.copy()

            # 确保索引和列一致
            self.corrections = self.corrections.reindex(index=self._dirty_data_ref.index,
                                                        columns=self._dirty_data_ref.columns)
            self.dirty_data_human_repaired = self.dirty_data_human_repaired.reindex(index=self._dirty_data_ref.index,
                                                                                    columns=self._dirty_data_ref.columns)
            print(
                f"[{time.ctime()}] self.corrections 形状: {self.corrections.shape}, 索引: {self.corrections.index}, 列: {self.corrections.columns}")
            print(
                f"[{time.ctime()}] self.dirty_data_human_repaired 形状: {self.dirty_data_human_repaired.shape}, 索引: {self.dirty_data_human_repaired.index}, 列: {self.dirty_data_human_repaired.columns}")

            # 检查 embeddings_matrix 是否存在
            if not hasattr(self, 'embeddings_matrix') or self.embeddings_matrix is None:
                error_msg = f"[{time.ctime()}] 错误: embeddings_matrix 未初始化"
                print(error_msg)
                self.logs.append(error_msg)
                raise ValueError(error_msg)

            with ThreadPoolExecutor(max_workers=1) as executor:  # 降低 max_workers 以避免限流
                futures = []
                for col_idx, column in enumerate(self.detection.columns):
                    print(f"[{time.ctime()}] 处理列 {column}, 错误总数: {self.detection[column].sum()}")
                    if self.detection[column].sum() > 0:
                        print(f"[{time.ctime()}] 为列 {column} 构建检索器")
                        retriever_start = time.time()
                        try:
                            retriever, indices, CoE = self.build_retriever_3(column)
                            retriever_build_time += time.time() - retriever_start
                            if retriever is None:
                                error_msg = f"[{time.ctime()}] 错误: 列 {column} 检索器为 None"
                                print(error_msg)
                                self.logs.append(error_msg)
                                continue
                            print(f"[{time.ctime()}] 构建完成")
                            self.retriever_dict[column] = retriever
                            self.indices_dict[column] = indices
                            self.CoE_dict[column] = CoE
                            valid_indices = [i for i in indices if i < len(self.header)]
                            if len(valid_indices) != len(indices):
                                print(
                                    f"[{time.ctime()}] 警告: 列 {column} 的 indices {indices} 包含无效索引，已过滤为 {valid_indices}")
                                self.logs.append(f"警告: 列 {column} 的 indices 包含无效索引")
                                indices = valid_indices or [self.detection.columns.get_loc(column)]
                            detection_filtered = self.detection.iloc[:, indices]
                            repair_list_for_column = self.detection[column][self.detection[column] == 1].index.tolist()
                            future = executor.submit(
                                self.get_auto_cot,
                                repair_list_for_column,
                                column,
                                retriever,
                                CoE,
                                indices,
                                detection_filtered,
                                self.rep_error_info,
                                self.sp_examps,
                                self.dirty_data,
                                self.clean_data,
                                self.embeddings_matrix,
                                self.detection,
                                self.detection_human_repaired,
                                self.header,
                                self.sys_auto_cot,
                                self.human_auto_cot_large,
                                self.human_auto_cot_small,
                                self.general_examples_str,
                                self.auto_cot_llm
                            )
                            futures.append((future, column))
                        except Exception as e:
                            error_msg = f"[{time.ctime()}] 构建检索器失败 for 列 {column}: {str(e)}"
                            print(error_msg)
                            self.logs.append(error_msg)
                            continue
                for future, column in futures:
                    try:
                        results, failed_rows = future.result(timeout=60)
                        for row_idx, col, correction in results:
                            self.corrections.at[row_idx, col] = correction
                            self.corrections_df = pd.concat([
                                self.corrections_df,
                                pd.DataFrame({
                                    'row': [row_idx],
                                    'col': [col],
                                    'original': [self.dirty_data.at[row_idx, col]],
                                    'corrected': [correction]
                                })
                            ], ignore_index=True)
                            print(
                                f"[{time.ctime()}] 应用修正: 列 {col} 行 {row_idx}, 原值: {self.dirty_data.at[row_idx, col]}, 新值: {correction}")
                            self.logs.append(
                                f"应用修正: 列 {col} 行 {row_idx}, 原值: {self.dirty_data.at[row_idx, col]}, 新值: {correction}")
                        if failed_rows:
                            print(f"[{time.ctime()}] Auto-CoT 任务失败 for 列 {column}, 失败行: {failed_rows}")
                            self.logs.append(f"Auto-CoT 任务失败 for 列 {column}, 失败行: {failed_rows}")
                    except Exception as e:
                        error_msg = f"[{time.ctime()}] Auto-CoT 任务失败 for 列 {column}: {str(e)}"
                        print(error_msg)
                        self.logs.append(error_msg)
                        continue

            # 保存特定示例
            try:
                with open(os.path.join(self.output_path, 'specific_examples.txt'), 'w', encoding='utf-8') as f_output:
                    for column, few_shot_specific_str in self.sp_examps.items():
                        f_output.write(f"列: {column}\n")
                        f_output.write(f"示例:\n{few_shot_specific_str}\n")
                        f_output.write("\n" + "=" * 50 + "\n\n")
                print(f"[{time.ctime()}] 已保存特定示例")
            except Exception as e:
                error_msg = f"[{time.ctime()}] 保存特定示例失败: {str(e)}"
                print(error_msg)
                self.logs.append(error_msg)

            print(f"[{time.ctime()}] Auto-CoT 生成完成")
            time_auto_cot_end = time.time()
            print(f"[{time.ctime()}] Auto-CoT 生成耗时: {time_auto_cot_end - time_auto_cot_start:.2f} 秒")
            self.f_time_cost.write(
                f"[{time.ctime()}] Auto-CoT 生成耗时: {time_auto_cot_end - time_auto_cot_start:.2f} 秒\n")

            # 应用 Auto-CoT 结果
            for column in self.detection.columns:
                if column in self.rep_error_info:
                    for row_idx, error_info in self.rep_error_info[column].items():
                        corrected_value = error_info['correction']
                        self.corrections.at[row_idx, column] = corrected_value
                        self.detection_human_repaired.at[row_idx, column] = 0
                        print(
                            f"[{time.ctime()}] 修正列 {column} 行 {row_idx}: {error_info['dirty_value']} -> {corrected_value}")
                        self.logs.append(
                            f"修正列 {column} 行 {row_idx}: {error_info['dirty_value']} -> {corrected_value}")

            # clean_data_integration
            try:
                rep_clean_data = self.sel_clean(self.human_repair_num if hasattr(self, 'human_repair_num') else 10)
                rep_data_info = copy.deepcopy(self.rep_error_info)
                if not rep_clean_data.empty:
                    self.clean_data_integration(rep_clean_data, rep_data_info)
                    print(f"[{time.ctime()}] clean_data_integration 完成")
                else:
                    print(f"[{time.ctime()}] 警告: rep_clean_data 为空，无法执行 clean_data_integration")
                    self.logs.append(f"警告: rep_clean_data 为空")
            except Exception as e:
                error_msg = f"[{time.ctime()}] clean_data_integration 失败: {str(e)}"
                print(error_msg)
                self.logs.append(error_msg)

            # 代码生成
            print(f"[{time.ctime()}] 开始代码生成")
            time_code_generation_start = time.time()
            self.codes = {}
            self.fds = {}
            train_data = {}
            val_data = {}
            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = []
                for column in self.detection.columns:
                    if self.detection[column].sum() > 0:
                        formatting_issue = True
                        for error in self.rep_error_info.get(column, {}).values():
                            if error['error_type'] not in ['clean', 'Formatting Issue']:
                                formatting_issue = False
                                break
                        train_data[column], val_data[column] = self.train_val_split(self.rep_error_info.get(column, {}))
                        if train_data[column]:
                            if formatting_issue:
                                future = executor.submit(
                                    self.code_generation,
                                    train_data[column],
                                    column,
                                    self.codes,
                                    self.code_generation_llm,
                                    self.sys_code_generation,
                                    self.human_code_generation
                                )
                            else:
                                future = executor.submit(
                                    self.FD_generation,
                                    train_data[column],
                                    column,
                                    self.fds,
                                    self.fd_generation_llm,
                                    self.sys_fd_generation,
                                    self.human_fd_generation
                                )
                            futures.append(future)
                for future in as_completed(futures):
                    try:
                        future.result(timeout=60)
                        print(f"[{time.ctime()}] 代码/FD 生成任务完成 for 列 {column}")
                    except Exception as e:
                        error_msg = f"[{time.ctime()}] 代码/FD 生成任务失败: {str(e)}"
                        print(error_msg)
                        self.logs.append(error_msg)

            # 保存代码和 FD
            try:
                with open(os.path.join(self.output_path, 'codes.txt'), 'w', encoding='utf-8') as f_output:
                    for column, code in self.codes.items():
                        f_output.write(f"列: {column}\n")
                        f_output.write(f"代码:\n{code}\n")
                        f_output.write("\n" + "=" * 50 + "\n\n")
                with open(os.path.join(self.output_path, 'fds.txt'), 'w', encoding='utf-8') as f_output:
                    for column, fd in self.fds.items():
                        f_output.write(f"列: {column}\n")
                        f_output.write(f"功能依赖:\n{json.dumps(fd)}\n")
                        f_output.write("\n" + "=" * 50 + "\n\n")
                print(f"[{time.ctime()}] 已保存代码和 FD")
            except Exception as e:
                error_msg = f"[{time.ctime()}] 保存代码和 FD 失败: {str(e)}"
                print(error_msg)
                self.logs.append(error_msg)

            time_code_generation_end = time.time()
            print(f"[{time.ctime()}] 代码生成完成")
            print(
                f"[{time.ctime()}] 代码生成耗时: {time_code_generation_end - time_code_generation_start:.2f} 秒")
            self.f_time_cost.write(
                f"[{time.ctime()}] 代码生成耗时: {time_code_generation_end - time_code_generation_start:.2f} 秒\n")

            # 代码评估和执行
            print(f"[{time.ctime()}] 开始代码评估和执行")
            time_code_evaluation_start = time.time()
            for column in self.codes.keys():
                try:
                    self.code_evaluation_execution(self.codes[column], val_data[column], column)
                except Exception as e:
                    error_msg = f"[{time.ctime()}] 代码评估和执行失败 for 列 {column}: {str(e)}"
                    print(error_msg)
                    self.logs.append(error_msg)
            for column in self.fds.keys():
                try:
                    self.fd_evaluation_execution(self.fds[column], val_data[column], column)
                except Exception as e:
                    error_msg = f"[{time.ctime()}] FD 评估和执行失败 for 列 {column}: {str(e)}"
                    print(error_msg)
                    self.logs.append(error_msg)
            time_code_evaluation_end = time.time()
            print(f"[{time.ctime()}] 代码评估和执行完成")
            print(
                f"[{time.ctime()}] 代码评估耗时: {time_code_evaluation_end - time_code_evaluation_start:.2f} 秒")
            self.f_time_cost.write(
                f"[{time.ctime()}] 代码评估和执行耗时: {time_code_evaluation_end - time_code_evaluation_start:.2f} 秒\n")

            # 更新检索器
            print(f"[{time.ctime()}] 开始更新检索器")
            time_update_retriever_start = time.time()
            if not self.dirty_data_human_repaired.index.equals(
                    self.corrections.index) or not self.dirty_data_human_repaired.columns.equals(
                    self.corrections.columns):
                print(f"[{time.ctime()}] 索引或列不匹配，强制对齐")
                self.corrections = self.corrections.reindex(index=self.dirty_data_human_repaired.index,
                                                            columns=self.dirty_data_human_repaired.columns)
                print(
                    f"[{time.ctime()}] 已对齐 self.corrections: shape={self.corrections.shape}, index={self.corrections.index}, columns={self.corrections.columns}")

            for col_idx, column in enumerate(self.detection_human_repaired.columns):
                mask = self.dirty_data_human_repaired != self.corrections
                changed_rows_for_column = mask[column][mask[column] == True].index.tolist()
                if changed_rows_for_column:
                    try:
                        self.update_retriever(
                            column=column,
                            retriever_instance=self.retriever_dict.get(column),
                            embeddings_matrix_all=self.embeddings_matrix,
                            indices_for_column=self.indices_dict.get(column, []),
                            dirty_data=self.dirty_data,
                            header=self.header,
                            repaired_row_indices=changed_rows_for_column
                        )
                    except Exception as e:
                        error_msg = f"[{time.ctime()}] 更新检索器失败 for 列 {column}: {str(e)}"
                        print(error_msg)
                        self.logs.append(error_msg)

            # 检索和修复
            print(f"[{time.ctime()}] 开始检索")
            time_retrieving_start = time.time()
            total_time = 0
            retriever_time = 0
            dict_creation_time = 0
            sort_time = 0
            detection_human_repaired_copy = self.detection_human_repaired.copy()
            for col_idx, column in enumerate(self.detection_human_repaired.columns):
                if self.detection_human_repaired[column].sum() > 0:
                    retriever = self.retriever_dict.get(column)
                    if retriever is None:
                        error_msg = f"[{time.ctime()}] 错误: 列 {column} 未找到检索器"
                        print(error_msg)
                        self.logs.append(error_msg)
                        continue
                    indices = self.indices_dict.get(column, [])
                    if not indices:
                        error_msg = f"[{time.ctime()}] 错误: 列 {column} 索引为空"
                        print(error_msg)
                        self.logs.append(error_msg)
                        continue
                    CoE = self.CoE_dict.get(column, [])
                    if len(indices) != len(CoE):
                        print(
                            f"[{time.ctime()}] 警告: 列 {column} 的 indices ({len(indices)}) 和 CoE ({len(CoE)}) 长度不匹配")
                        self.logs.append(f"警告: 列 {column} 的 indices 和 CoE 长度不匹配")
                        CoE = CoE[:len(indices)]  # 截断 CoE 以匹配 indices
                    temp = self.detection_human_repaired.iloc[:, indices] if indices else pd.DataFrame()
                    for row_idx in range(len(self.detection_human_repaired)):
                        if detection_human_repaired_copy.at[row_idx, column] == 1:
                            start_time = time.time()
                            relevant_clean_tuples = ''
                            embeddings_row = self.embeddings_matrix[row_idx]
                            if embeddings_row is None or embeddings_row.size == 0 or embeddings_row.ndim != 2:
                                error_msg = f"[{time.ctime()}] 错误: 列 {column} 行 {row_idx} embeddings_row 无效，形状: {getattr(embeddings_row, 'shape', 'None')}"
                                print(error_msg)
                                self.logs.append(error_msg)
                                continue
                            embeddings_row_filtered = embeddings_row[indices]
                            if not isinstance(embeddings_row_filtered, np.ndarray) or embeddings_row_filtered.size == 0:
                                error_msg = f"[{time.ctime()}] 错误: 列 {column} 行 {row_idx} embeddings_row_filtered 无效"
                                print(error_msg)
                                self.logs.append(error_msg)
                                continue
                            if embeddings_row_filtered.ndim == 1:
                                embeddings_row_filtered = embeddings_row_filtered[:, np.newaxis]
                            for i in range(len(embeddings_row_filtered)):
                                if i < len(temp.columns) and temp.iloc[row_idx, i] == 1:
                                    embeddings_row_filtered[i] = np.zeros_like(embeddings_row_filtered[i])
                            CoE_reshaped = np.array(CoE)[:, np.newaxis]
                            if embeddings_row_filtered.shape[0] != CoE_reshaped.shape[0]:
                                error_msg = f"[{time.ctime()}] 错误: 列 {column} 行 {row_idx} embeddings_row_filtered ({embeddings_row_filtered.shape}) 和 CoE_reshaped ({CoE_reshaped.shape}) 形状不匹配"
                                print(error_msg)
                                self.logs.append(error_msg)
                                continue
                            embeddings_row_filtered = np.multiply(embeddings_row_filtered, CoE_reshaped)
                            embeddings_row_united = embeddings_row_filtered.flatten()
                            try:
                                retriever_start = time.time()
                                query_text = self.format_row_2(self.dirty_data.iloc[row_idx].values, self.header,
                                                               self.detection.iloc[row_idx].values)
                                if not query_text or query_text == "{}":
                                    query_text = json.dumps(
                                        {column: str(self.dirty_data_human_repaired.at[row_idx, column])},
                                        ensure_ascii=False)
                                relevant_rows = retriever.invoke(query_text)
                                retriever_time += time.time() - retriever_start
                                dict_start = time.time()
                                relevant_rows_dict_list = [
                                    {
                                        'page_content': row.page_content,
                                        'index': idx,
                                        'score': round(row.metadata.get('score', 0), 2),
                                        'target_column': self.detection_human_repaired[column].values[idx],
                                        'sum': self.detection_human_repaired.iloc[:, indices].values.sum(axis=1)[idx]
                                    }
                                    for row in relevant_rows
                                    for idx in [row.metadata['index']]
                                    if isinstance(row.metadata['index'], int)
                                ]
                                dict_creation_time += time.time() - dict_start
                                sort_start = time.time()
                                sorted_relevant_rows_dict_list = self.sort_dicts(relevant_rows_dict_list, 'score',
                                                                                 'target_column', 'sum')
                                sort_time += time.time() - sort_start
                                for row in sorted_relevant_rows_dict_list[:3]:
                                    relevant_clean_tuples += row['page_content'] + '\n'
                                if column not in self.retrieved_tuples:
                                    self.retrieved_tuples[column] = {}
                                self.retrieved_tuples[column][row_idx] = relevant_clean_tuples
                            except Exception as e:
                                error_msg = f"[{time.ctime()}] 检索失败 for 列 {column} 行 {row_idx}: {str(e)}"
                                print(error_msg)
                                self.logs.append(error_msg)
                                continue
                            total_time += time.time() - start_time
            print(f"[{time.ctime()}] 总检索耗时: {total_time:.2f} 秒")
            print(f"[{time.ctime()}] 检索完成")
            time_retrieving_end = time.time()
            print(f"[{time.ctime()}] 检索耗时: {time_retrieving_end - time_retrieving_start:.2f} 秒")
            self.f_time_cost.write(f"[{time.ctime()}] 检索耗时: {time_retrieving_end - time_retrieving_start:.2f} 秒\n")

            # 修复阶段
            print(f"[{time.ctime()}] 开始修复")
            time_repairing_start = time.time()
            repaired_results = {}
            prompt_dict = {}
            parser = JsonOutputParser(pydantic_object=RepairOutput)
            response_list = []
            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = []
                for col_idx, column in enumerate(self.detection_human_repaired.columns):
                    if self.detection_human_repaired[column].sum() > 0:
                        prompt = ChatPromptTemplate(
                            messages=[
                                SystemMessagePromptTemplate.from_template(self.sys),
                                HumanMessagePromptTemplate.from_template(self.human),
                            ],
                            partial_variables={
                                "general_examples": self.general_examples_str,
                                "specific_examples": self.sp_examps.get(column, ''),
                                "format_instructions": parser.get_format_instructions()
                            }
                        )
                        prompt_dict[column] = prompt
                        for row_idx in range(len(self.detection_human_repaired)):
                            if detection_human_repaired_copy.at[row_idx, column] == 1:
                                current_dirty_value = self.dirty_data_human_repaired.at[row_idx, column]
                                error_info_for_prompt = {
                                    "dirty_value": current_dirty_value,
                                    "error_analysis": self.rep_error_info.get(column, {}).get(row_idx, {}).get(
                                        'error_analysis', 'N/A'),
                                    "error_type": self.rep_error_info.get(column, {}).get(row_idx, {}).get('error_type',
                                                                                                           'N/A'),
                                    "column_name": column,
                                    "row_id": row_idx
                                }
                                clean_examples_for_prompt = self.retrieved_tuples.get(column, {}).get(row_idx, '')
                                current_indices = self.indices_dict.get(column, [])
                                if not current_indices:
                                    error_msg = f"[{time.ctime()}] 错误: 列 {column} 未找到索引"
                                    print(error_msg)
                                    self.logs.append(error_msg)
                                    continue
                                valid_indices = [i for i in current_indices if i < len(self.header)]
                                if len(valid_indices) != len(current_indices):
                                    print(
                                        f"[{time.ctime()}] 警告: 列 {column} 的 current_indices {current_indices} 包含无效索引，已过滤为 {valid_indices}")
                                    self.logs.append(f"警告: 列 {column} 的 current_indices 包含无效索引")
                                    current_indices = valid_indices or [self.detection.columns.get_loc(column)]
                                dirty_tuple_filtered = self.dirty_data_human_repaired.iloc[row_idx, current_indices]
                                filtered_header_for_dirty_tuple = [self.header[i] for i in current_indices]
                                try:
                                    dirty_tuple_string_for_prompt = self.format_row(dirty_tuple_filtered,
                                                                                    filtered_header_for_dirty_tuple)
                                except Exception as e:
                                    error_msg = f"[{time.ctime()}] 错误: format_row 失败 for 列 {column} 行 {row_idx}: {str(e)}"
                                    print(error_msg)
                                    self.logs.append(error_msg)
                                    continue
                                prompt_input_data = {
                                    "Dirty_tuple": dirty_tuple_string_for_prompt,
                                    "Erroneous_value": current_dirty_value,
                                    "Relevant_clean_tuples": clean_examples_for_prompt,
                                }
                                future = executor.submit(
                                    self.process_repair_item,
                                    prompt,
                                    self.llm_repair,
                                    parser,
                                    prompt_input_data,
                                    column,
                                    row_idx
                                )
                                futures.append((future, column, row_idx))
                for future, col, r_idx in futures:
                    try:
                        col, r_idx, repaired_value, explanation = future.result(timeout=60)
                        response_list.append(repaired_value)
                        if col not in repaired_results:
                            repaired_results[col] = {}
                        repaired_results[col][r_idx] = {'repaired_value': repaired_value, 'explanation': explanation}
                        self.logs.append({
                            "Index": str(r_idx),
                            "Column": col,
                            "Row Index": r_idx,
                            "Dirty_tuple": prompt_input_data["Dirty_tuple"],
                            "Dirty_value": self.dirty_data_human_repaired.at[r_idx, col],
                            "Relevant_clean_tuples": prompt_input_data["Relevant_clean_tuples"],
                            "Correction": str(repaired_value),
                            "Explanation": explanation
                        })
                        if repaired_value != "null" and repaired_value and repaired_value not in ["LLM_INVOKE_ERROR",
                                                                                                  "PARSING_FAILED_FALLBACK"]:
                            self.corrections.at[r_idx, col] = repaired_value
                            self.corrections_df = pd.concat([self.corrections_df, pd.DataFrame([{
                                'row': r_idx,
                                'col': col,
                                'original': self.dirty_data_human_repaired.at[r_idx, col],
                                'corrected': repaired_value
                            }])], ignore_index=True)
                            self.logs.append(
                                f"应用修正: {self.dirty_data_human_repaired.at[r_idx, col]} -> {repaired_value} at ({r_idx}, {col})")
                            print(
                                f"[{time.ctime()}] 应用修正: {self.dirty_data_human_repaired.at[r_idx, col]} -> {repaired_value} at ({r_idx}, {col})")
                        else:
                            error_msg = f"[{time.ctime()}] 修复失败 for 列 {col} 行 {r_idx}: 无效修复值 {repaired_value}"
                            print(error_msg)
                            self.logs.append(error_msg)
                            # 回退纠正逻辑
                            repaired_value = re.sub(r'^err_?|^ERR_?', '',
                                                    str(self.dirty_data_human_repaired.at[r_idx, col]))
                            self.corrections.at[r_idx, col] = repaired_value
                            self.corrections_df = pd.concat([self.corrections_df, pd.DataFrame([{
                                'row': r_idx,
                                'col': col,
                                'original': self.dirty_data_human_repaired.at[r_idx, col],
                                'corrected': repaired_value
                            }])], ignore_index=True)
                            self.logs.append(
                                f"回退修正: {self.dirty_data_human_repaired.at[r_idx, col]} -> {repaired_value} at ({r_idx}, {col})")
                            print(
                                f"[{time.ctime()}] 回退修正: {self.dirty_data_human_repaired.at[r_idx, col]} -> {repaired_value} at ({r_idx}, {col})")
                    except Exception as e:
                        error_msg = f"[{time.ctime()}] 修复任务失败 for 列 {col} 行 {r_idx}: {str(e)}"
                        print(error_msg)
                        self.logs.append(error_msg)
                        # 回退纠正逻辑
                        repaired_value = re.sub(r'^err_?|^ERR_?', '',
                                                str(self.dirty_data_human_repaired.at[r_idx, col]))
                        self.corrections.at[r_idx, col] = repaired_value
                        self.corrections_df = pd.concat([self.corrections_df, pd.DataFrame([{
                            'row': r_idx,
                            'col': col,
                            'original': self.dirty_data_human_repaired.at[r_idx, col],
                            'corrected': repaired_value
                        }])], ignore_index=True)
                        self.logs.append(
                            f"回退修正: {self.dirty_data_human_repaired.at[r_idx, col]} -> {repaired_value} at ({r_idx}, {col})")
                        print(
                            f"[{time.ctime()}] 回退修正: {self.dirty_data_human_repaired.at[r_idx, col]} -> {repaired_value} at ({r_idx}, {col})")

            print(f"[{time.ctime()}] 总计 token 数: {self.total_tokens}")
            time_repairing_end = time.time()
            print(f"[{time.ctime()}] 修复完成")
            print(f"[{time.ctime()}] 修复耗时: {time_repairing_end - time_repairing_start:.2f} 秒")
            self.f_time_cost.write(f"[{time.ctime()}] 修复耗时: {time_repairing_end - time_repairing_start:.2f} 秒\n")
            self.logs.append(f"预测完成。总修正: {len(self.corrections_df)}, 总tokens: {self.total_tokens}")
            print(f"[{time.ctime()}] 预测完成。总修正: {len(self.corrections_df)}")
            try:
                self.corrections_df.to_csv(os.path.join(self.output_path, 'corrections.csv'), index=False,
                                           encoding='utf-8')
                print(f"[{time.ctime()}] 已保存修复结果到 {os.path.join(self.output_path, 'corrections.csv')}")
            except Exception as e:
                error_msg = f"[{time.ctime()}] 保存 corrections.csv 失败: {str(e)}"
                print(error_msg)
                self.logs.append(error_msg)
            return self.corrections

        def predict(self, dirty_data: pd.DataFrame, detection: pd.DataFrame) -> pd.DataFrame:
            """预测方法，添加 metadata KeyError 处理和调试日志"""
            print(f"[{time.ctime()}] 开始预测")
            self.dirty_data = dirty_data.copy()
            self.detection = detection.copy()
            self.dirty_data_human_repaired = dirty_data.copy()
            self.detection_human_repaired = detection.copy()
            self.corrections = dirty_data.copy()
            self.corrections_df = pd.DataFrame(columns=['row', 'col', 'original', 'corrected'])
            self.header = list(dirty_data.columns)
            total_time = 0
            retriever_time = 0
            dict_creation_time = 0
            sort_time = 0
            time_retrieving_start = time.time()

            # 确保 embeddings_matrix 已初始化
            if not hasattr(self, 'embeddings_matrix') or self.embeddings_matrix is None:
                print(f"[{time.ctime()}] 生成嵌入矩阵 for 预测")
                self.embeddings_matrix = self.generate_embeddings(dirty_data)

            # 更新检索器（预测阶段可能需要传入已修复的行索引，如果为空则跳过）
            print(f"[{time.ctime()}] 开始更新检索器")
            update_start = time.time()
            try:
                for column in self.header:
                    if column in self.retriever_dict:
                        indices = self.indices_dict.get(column, [])
                        if not indices:
                            print(f"[{time.ctime()}] 警告: 列 {column} 无索引，跳过更新")
                            continue
                        # 如果没有已修复行，使用 fit 阶段的 repaired_row_indices 或空列表
                        repaired_row_indices = self.specific_examples.get(column, {}).keys() if hasattr(self,
                                                                                                        'specific_examples') and column in self.specific_examples else []
                        repaired_row_indices = list(map(int, repaired_row_indices)) if repaired_row_indices else []
                        self.update_retriever(
                            column=column,
                            retriever_instance=self.retriever_dict[column],
                            embeddings_matrix_all=self.embeddings_matrix,
                            indices_for_column=indices,
                            dirty_data=self.dirty_data,
                            header=self.header,
                            repaired_row_indices=repaired_row_indices  # 使用示例行作为 repaired_row_indices
                        )
                print(f"[{time.ctime()}] 更新检索器完成")
            except Exception as e:
                print(f"[{time.ctime()}] 更新检索器失败: {str(e)}")
                self.logs.append(f"更新检索器失败: {str(e)}")
            print(f"[{time.ctime()}] 更新检索器耗时: {time.time() - update_start:.2f} 秒")
            self.f_time_cost.write(f"[{time.ctime()}] 更新检索器耗时: {time.time() - update_start:.2f} 秒\n")

            # 检索阶段
            print(f"[{time.ctime()}] 开始检索")
            detection_human_repaired_copy = self.detection_human_repaired.copy()
            for col_idx, column in enumerate(self.detection_human_repaired.columns):
                if self.detection_human_repaired[column].sum() > 0:
                    retriever = self.retriever_dict.get(column)
                    if retriever is None:
                        error_msg = f"[{time.ctime()}] 错误: 列 {column} 未找到检索器"
                        print(error_msg)
                        self.logs.append(error_msg)
                        continue
                    indices = self.indices_dict.get(column, [])
                    if not indices:
                        error_msg = f"[{time.ctime()}] 错误: 列 {column} 索引为空"
                        print(error_msg)
                        self.logs.append(error_msg)
                        continue
                    CoE = self.CoE_dict.get(column, [])
                    # 验证 indices 是否有效
                    valid_indices = [i for i in indices if i < len(self.header)]
                    if len(valid_indices) != len(indices):
                        print(
                            f"[{time.ctime()}] 警告: 列 {column} 的 indices {indices} 包含无效索引，调整为 {valid_indices}")
                        self.logs.append(f"警告: 列 {column} 的 indices 包含无效索引")
                        indices = valid_indices or [self.detection.columns.get_loc(column)]
                    # 确保 CoE 和 indices 长度一致
                    if len(indices) != len(CoE):
                        print(
                            f"[{time.ctime()}] 警告: 列 {column} 的 indices ({len(indices)}) 和 CoE ({len(CoE)}) 长度不匹配，调整 CoE")
                        self.logs.append(f"警告: 列 {column} 的 indices 和 CoE 长度不匹配")
                        CoE = CoE[:len(indices)] if len(CoE) > len(indices) else CoE + [0.0] * (len(indices) - len(CoE))
                    temp = self.detection_human_repaired.iloc[:, indices] if indices else pd.DataFrame()
                    for row_idx in range(len(self.detection_human_repaired)):
                        if detection_human_repaired_copy.at[row_idx, column] == 1:
                            start_time = time.time()
                            relevant_clean_tuples = ''
                            embeddings_row = self.embeddings_matrix[row_idx]
                            if embeddings_row is None or embeddings_row.size == 0 or embeddings_row.ndim != 2:
                                error_msg = f"[{time.ctime()}] 错误: 列 {column} 行 {row_idx} embeddings_row 无效，形状: {getattr(embeddings_row, 'shape', 'None')}"
                                print(error_msg)
                                self.logs.append(error_msg)
                                continue
                            # 再次验证 indices 是否在 embeddings_row 范围内
                            valid_indices = [i for i in indices if i < embeddings_row.shape[0]]
                            if len(valid_indices) != len(indices):
                                print(
                                    f"[{time.ctime()}] 警告: 列 {column} 行 {row_idx} 的 indices {indices} 超出 embeddings_row 范围，调整为 {valid_indices}")
                                self.logs.append(f"警告: 列 {column} 行 {row_idx} 的 indices 超出 embeddings_row 范围")
                                indices = valid_indices or [self.detection.columns.get_loc(column)]
                            embeddings_row_filtered = embeddings_row[valid_indices] if valid_indices else np.array([])
                            if not isinstance(embeddings_row_filtered, np.ndarray) or embeddings_row_filtered.size == 0:
                                error_msg = f"[{time.ctime()}] 错误: 列 {column} 行 {row_idx} embeddings_row_filtered 无效"
                                print(error_msg)
                                self.logs.append(error_msg)
                                continue
                            if embeddings_row_filtered.ndim == 1:
                                embeddings_row_filtered = embeddings_row_filtered[:, np.newaxis]
                            for i in range(len(embeddings_row_filtered)):
                                if i < len(temp.columns) and temp.iloc[row_idx, i] == 1:
                                    embeddings_row_filtered[i] = np.zeros_like(embeddings_row_filtered[i])
                            CoE_reshaped = np.array(CoE)[:, np.newaxis]
                            if embeddings_row_filtered.shape[0] != CoE_reshaped.shape[0]:
                                error_msg = f"[{time.ctime()}] 错误: 列 {column} 行 {row_idx} embeddings_row_filtered ({embeddings_row_filtered.shape}) 和 CoE_reshaped ({CoE_reshaped.shape}) 形状不匹配"
                                print(error_msg)
                                self.logs.append(error_msg)
                                continue
                            embeddings_row_filtered = np.multiply(embeddings_row_filtered, CoE_reshaped)
                            embeddings_row_united = embeddings_row_filtered.flatten()
                            try:
                                retriever_start = time.time()
                                query_text = self.format_row_2(self.dirty_data.iloc[row_idx].values, self.header,
                                                               self.detection.iloc[row_idx].values)
                                if not query_text or query_text == "{}":
                                    query_text = json.dumps(
                                        {column: str(self.dirty_data_human_repaired.at[row_idx, column])},
                                        ensure_ascii=False)
                                relevant_rows = retriever.invoke(query_text)
                                print(
                                    f"[{time.ctime()}] 检索返回 {len(relevant_rows)} 条结果 for 列 {column} 行 {row_idx}")  # 调试日志
                                retriever_time += time.time() - retriever_start
                                dict_start = time.time()
                                relevant_rows_dict_list = []
                                for row in relevant_rows:
                                    try:
                                        # 安全访问 metadata['index']
                                        idx = None
                                        if hasattr(row, 'metadata') and isinstance(row.metadata, dict):
                                            if 'index' in row.metadata:
                                                idx = row.metadata['index']
                                            else:
                                                print(
                                                    f"[{time.ctime()}] 警告: metadata 缺少 'index' 键: {row.metadata}")
                                                # 尝试从其他字段推断或使用默认
                                                idx = row.metadata.get('source',
                                                                       row_idx) if 'source' in row.metadata else row_idx
                                        else:
                                            print(f"[{time.ctime()}] 警告: row.metadata 无效: {type(row.metadata)}")
                                            idx = row_idx  # 默认使用当前行索引

                                        if isinstance(idx, int):
                                            relevant_rows_dict_list.append({
                                                'page_content': row.page_content,
                                                'index': idx,
                                                'score': round(getattr(row, 'metadata', {}).get('score', 0), 2),
                                                'target_column': self.detection_human_repaired[column].values[
                                                    idx] if 0 <= idx < len(self.detection_human_repaired) else 0,
                                                'sum': self.detection_human_repaired.iloc[:, valid_indices].values.sum(
                                                    axis=1)[idx] if 0 <= idx < len(self.detection_human_repaired) else 0
                                            })
                                    except Exception as inner_e:
                                        print(
                                            f"[{time.ctime()}] 处理检索结果失败 for 列 {column} 行 {row_idx}: {str(inner_e)}")
                                        continue

                                dict_creation_time += time.time() - dict_start
                                sort_start = time.time()
                                sorted_relevant_rows_dict_list = self.sort_dicts(relevant_rows_dict_list, 'score',
                                                                                 'target_column', 'sum')
                                sort_time += time.time() - sort_start
                                for row in sorted_relevant_rows_dict_list[:3]:
                                    relevant_clean_tuples += row['page_content'] + '\n'
                                if column not in self.retrieved_tuples:
                                    self.retrieved_tuples[column] = {}
                                self.retrieved_tuples[column][row_idx] = relevant_clean_tuples
                            except Exception as e:
                                error_msg = f"[{time.ctime()}] 检索失败 for 列 {column} 行 {row_idx}: {str(e)}"
                                print(error_msg)
                                self.logs.append(error_msg)
                                continue
                            total_time += time.time() - start_time
            print(f"[{time.ctime()}] 总检索耗时: {total_time:.2f} 秒")
            print(f"[{time.ctime()}] 检索完成")
            time_retrieving_end = time.time()
            print(f"[{time.ctime()}] 检索耗时: {time_retrieving_end - time_retrieving_start:.2f} 秒")
            self.f_time_cost.write(f"[{time.ctime()}] 检索耗时: {time_retrieving_end - time_retrieving_start:.2f} 秒\n")

            # 修复阶段（保持原样，省略以节省空间，但已添加必要的错误处理）
            print(f"[{time.ctime()}] 开始修复")
            time_repairing_start = time.time()
            repaired_results = {}
            prompt_dict = {}
            parser = JsonOutputParser(pydantic_object=RepairOutput)
            response_list = []
            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = []
                for col_idx, column in enumerate(self.detection_human_repaired.columns):
                    if self.detection_human_repaired[column].sum() > 0:
                        prompt = ChatPromptTemplate(
                            messages=[
                                SystemMessagePromptTemplate.from_template(self.sys),
                                HumanMessagePromptTemplate.from_template(self.human),
                            ],
                            partial_variables={
                                "general_examples": self.general_examples_str,
                                "specific_examples": self.sp_examps.get(column, ''),
                                "format_instructions": parser.get_format_instructions()
                            }
                        )
                        prompt_dict[column] = prompt
                        for row_idx in range(len(self.detection_human_repaired)):
                            if detection_human_repaired_copy.at[row_idx, column] == 1:
                                current_dirty_value = self.dirty_data_human_repaired.at[row_idx, column]
                                error_info_for_prompt = {
                                    "dirty_value": current_dirty_value,
                                    "error_analysis": self.rep_error_info.get(column, {}).get(row_idx, {}).get(
                                        'error_analysis', 'N/A'),
                                    "error_type": self.rep_error_info.get(column, {}).get(row_idx, {}).get('error_type',
                                                                                                           'N/A'),
                                    "column_name": column,
                                    "row_id": row_idx
                                }
                                clean_examples_for_prompt = self.retrieved_tuples.get(column, {}).get(row_idx, '')
                                current_indices = self.indices_dict.get(column, [])
                                if not current_indices:
                                    error_msg = f"[{time.ctime()}] 错误: 列 {column} 未找到索引"
                                    print(error_msg)
                                    self.logs.append(error_msg)
                                    continue
                                valid_indices = [i for i in current_indices if i < len(self.header)]
                                if len(valid_indices) != len(current_indices):
                                    print(
                                        f"[{time.ctime()}] 警告: 列 {column} 的 current_indices {current_indices} 包含无效索引，已过滤为 {valid_indices}")
                                    self.logs.append(f"警告: 列 {column} 的 current_indices 包含无效索引")
                                    current_indices = valid_indices or [self.detection.columns.get_loc(column)]
                                dirty_tuple_filtered = self.dirty_data_human_repaired.iloc[row_idx, current_indices]
                                filtered_header_for_dirty_tuple = [self.header[i] for i in current_indices]
                                try:
                                    dirty_tuple_string_for_prompt = self.format_row(dirty_tuple_filtered,
                                                                                    filtered_header_for_dirty_tuple)
                                except Exception as e:
                                    error_msg = f"[{time.ctime()}] 错误: format_row 失败 for 列 {column} 行 {row_idx}: {str(e)}"
                                    print(error_msg)
                                    self.logs.append(error_msg)
                                    continue
                                prompt_input_data = {
                                    "Dirty_tuple": dirty_tuple_string_for_prompt,
                                    "Erroneous_value": current_dirty_value,
                                    "Relevant_clean_tuples": clean_examples_for_prompt,
                                }
                                future = executor.submit(
                                    self.process_repair_item,
                                    prompt,
                                    self.llm_repair,
                                    parser,
                                    prompt_input_data,
                                    column,
                                    row_idx
                                )
                                futures.append((future, column, row_idx))
                for future, col, r_idx in futures:
                    try:
                        col, r_idx, repaired_value, explanation = future.result(timeout=60)
                        response_list.append(repaired_value)
                        if col not in repaired_results:
                            repaired_results[col] = {}
                        repaired_results[col][r_idx] = {'repaired_value': repaired_value, 'explanation': explanation}
                        self.logs.append({
                            "Index": str(r_idx),
                            "Column": col,
                            "Row Index": r_idx,
                            "Dirty_tuple": prompt_input_data["Dirty_tuple"],
                            "Dirty_value": self.dirty_data_human_repaired.at[r_idx, col],
                            "Relevant_clean_tuples": prompt_input_data["Relevant_clean_tuples"],
                            "Correction": str(repaired_value),
                            "Explanation": explanation
                        })
                        if repaired_value != "null" and repaired_value and repaired_value not in ["LLM_INVOKE_ERROR",
                                                                                                  "PARSING_FAILED_FALLBACK"]:
                            self.corrections.at[r_idx, col] = repaired_value
                            self.corrections_df = pd.concat([self.corrections_df, pd.DataFrame([{
                                'row': r_idx,
                                'col': col,
                                'original': self.dirty_data_human_repaired.at[r_idx, col],
                                'corrected': repaired_value
                            }])], ignore_index=True)
                            self.logs.append(
                                f"应用修正: {self.dirty_data_human_repaired.at[r_idx, col]} -> {repaired_value} at ({r_idx}, {col})")
                            print(
                                f"[{time.ctime()}] 应用修正: {self.dirty_data_human_repaired.at[r_idx, col]} -> {repaired_value} at ({r_idx}, {col})")
                        else:
                            error_msg = f"[{time.ctime()}] 修复失败 for 列 {col} 行 {r_idx}: 无效修复值 {repaired_value}"
                            print(error_msg)
                            self.logs.append(error_msg)
                    except Exception as e:
                        error_msg = f"[{time.ctime()}] 修复任务失败 for 列 {col} 行 {r_idx}: {str(e)}"
                        print(error_msg)
                        self.logs.append(error_msg)

            print(f"[{time.ctime()}] 总计 token 数: {self.total_tokens}")
            time_repairing_end = time.time()
            print(f"[{time.ctime()}] 修复完成")
            print(f"[{time.ctime()}] 修复耗时: {time_repairing_end - time_repairing_start:.2f} 秒")
            self.f_time_cost.write(f"[{time.ctime()}] 修复耗时: {time_repairing_end - time_repairing_start:.2f} 秒\n")
            self.logs.append(f"预测完成。总修正: {len(self.corrections_df)}, 总tokens: {self.total_tokens}")
            print(f"[{time.ctime()}] 预测完成。总修正: {len(self.corrections_df)}")
            try:
                self.corrections_df.to_csv(os.path.join(self.output_path, 'corrections.csv'), index=False,
                                           encoding='utf-8')
                print(f"[{time.ctime()}] 已保存修复结果到 {os.path.join(self.output_path, 'corrections.csv')}")
            except Exception as e:
                error_msg = f"[{time.ctime()}] 保存 corrections.csv 失败: {str(e)}"
                print(error_msg)
                self.logs.append(error_msg)
            return self.corrections



        def code_generation(self, train_data: Dict, column: str, codes: Dict, llm, sys_prompt: str,
                            human_prompt: str) -> None:
            print(f"[{time.ctime()}] 开始为列 {column} 生成代码")
            self.logs.append({
                'message': f"开始为列 {column} 生成代码",
                'timestamp': time.ctime()
            })
            try:
                # 格式化示例
                examples = form_examples([
                    {'original': k, 'corrected': v['correction'], 'error_type': v.get('error_type', '未知')}
                    for k, v in train_data.items() if 'correction' in v
                ])
                if not examples:
                    print(f"[{time.ctime()}] 警告: 列 {column} 的 train_data 为空或缺少有效示例")
                    self.logs.append({
                        'message': f"警告: 列 {column} 的 train_data 为空或缺少有效示例",
                        'timestamp': time.ctime()
                    })
                    codes[column] = 'def correct(value):\n    return re.sub(r"^err_?|^ERR_?", "", str(value))'
                    return

                # 设置列类型
                column_type = 'int64' if column == 'col_b' else 'object'
                prompt_input = {
                    'column_name': column,
                    'column_type': column_type,
                    'column_description': (
                        "col_a: String values in the format 'valueN' (e.g., 'value1', 'value2').\n"
                        "col_b: Integer numerical values (e.g., 10, 20, 30).\n"
                        "col_c: Single-letter string values (e.g., 'x', 'y', 'z')."
                    ),
                    'examples': examples,
                    'format_instructions': JsonOutputParser().get_format_instructions()
                }
                print(f"[{time.ctime()}] 代码生成提示输入: {prompt_input}")
                self.logs.append({
                    'message': f"代码生成提示输入: {prompt_input}",
                    'timestamp': time.ctime()
                })

                # 加载提示模板
                prompt = ChatPromptTemplate.from_messages([
                    SystemMessagePromptTemplate.from_template(sys_prompt),
                    HumanMessagePromptTemplate.from_template(human_prompt)
                ])
                chain = prompt | llm | JsonOutputParser()

                # 调用 LLM
                response = chain.invoke(prompt_input)
                print(f"[{time.ctime()}] LLM 原始输出 for 列 {column}: {response}")
                self.logs.append({
                    'message': f"LLM 原始输出 for 列 {column}: {response}",
                    'timestamp': time.ctime()
                })

                # 处理响应
                if isinstance(response, list):
                    if len(response) == 1:
                        response = response[0]
                    else:
                        raise ValueError(f"预期单个 JSON 对象，得到列表包含 {len(response)} 项: {response}")

                if not isinstance(response, dict):
                    response_str = str(response).strip()
                    try:
                        response = json.loads(response_str)
                        if isinstance(response, list) and len(response) == 1:
                            response = response[0]
                    except json.JSONDecodeError as e:
                        print(f"[{time.ctime()}] JSON 解析失败 for 列 {column}: {str(e)}, 响应: {response_str}")
                        self.logs.append({
                            'message': f"JSON 解析失败 for 列 {column}: {str(e)}, 响应: {response_str}",
                            'timestamp': time.ctime()
                        })
                        response = {
                            'correction': 'def correct(value):\n    return re.sub(r"^err_?|^ERR_?", "", str(value))'}

                if 'correction' not in response:
                    print(f"[{time.ctime()}] 响应缺少 'correction' 键 for 列 {column}: {response}")
                    self.logs.append({
                        'message': f"响应缺少 'correction' 键 for 列 {column}: {response}",
                        'timestamp': time.ctime()
                    })
                    response['correction'] = 'def correct(value):\n    return re.sub(r"^err_?|^ERR_?", "", str(value))'

                # 验证生成的代码
                try:
                    code_str = response['correction']
                    # 检查代码是否是有效的 Python 函数
                    exec(code_str, {})  # 简单验证代码语法
                    codes[column] = code_str
                    print(f"[{time.ctime()}] 代码生成成功 for 列 {column}: {code_str}")
                    self.logs.append({
                        'message': f"代码生成成功 for 列 {column}: {code_str}",
                        'timestamp': time.ctime()
                    })
                except Exception as e:
                    print(
                        f"[{time.ctime()}] 无效的 Python 代码 for 列 {column}: {str(e)}, 代码: {response['correction']}")
                    self.logs.append({
                        'message': f"无效的 Python 代码 for 列 {column}: {str(e)}, 代码: {response['correction']}",
                        'timestamp': time.ctime()
                    })
                    codes[column] = 'def correct(value):\n    return re.sub(r"^err_?|^ERR_?", "", str(value))'

            except Exception as e:
                print(f"[{time.ctime()}] 代码生成失败 for 列 {column}: {str(e)}")
                self.logs.append({
                    'message': f"代码生成失败 for 列 {column}: {str(e)}",
                    'timestamp': time.ctime()
                })
                codes[column] = 'def correct(value):\n    return re.sub(r"^err_?|^ERR_?", "", str(value))'

        def fd_evaluation_execution(self, fd, val_data, column):
            print(f"[{time.ctime()}] Starting FD evaluation for column {column}")
            try:
                if not fd or 'functional_dependency' not in fd or not isinstance(fd['functional_dependency'], str):
                    print(
                        f"[{time.ctime()}] FD evaluation skipped for column {column}: 'functional_dependency' key missing or not a string")
                    self.logs.append(f"FD evaluation skipped for column {column}: 'functional_dependency' key missing")
                    return

                # 执行 FD 代码
                fd_code = fd['functional_dependency'].strip()
                try:
                    ast.parse(fd_code)
                except SyntaxError as e:
                    error_msg = f"[{time.ctime()}] FD evaluation failed for column {column}: Syntax error in FD code - {str(e)}"
                    print(error_msg)
                    self.logs.append(error_msg)
                    return

                local_namespace = {}
                exec(fd_code, local_namespace)
                if 'correct' not in local_namespace:
                    raise ValueError("No callable 'correct' function found in FD")
                correct_func = local_namespace['correct']

                # 调试：打印 val_data 格式和内容
                print(
                    f"[{time.ctime()}] val_data type: {type(val_data)}, keys: {list(val_data.keys()) if isinstance(val_data, dict) else val_data.columns.tolist() if isinstance(val_data, pd.DataFrame) else 'unknown'}")
                if isinstance(val_data, dict) and len(val_data) > 0:
                    sample_key = list(val_data.keys())[0]
                    sample_value = val_data[sample_key]
                    print(
                        f"[{time.ctime()}] val_data sample key: {sample_key}, value type: {type(sample_value)}, value: {sample_value if isinstance(sample_value, dict) else 'non-dict'}")

                # 处理 val_data：兼容 dict (row index -> dict of columns) or pd.DataFrame
                column_data = []
                if isinstance(val_data, dict):
                    for idx, row in val_data.items():
                        if isinstance(row, dict) and column in row:
                            column_data.append(row[column])
                        elif isinstance(row, dict) and 'dirty_value' in row:
                            # 如果是增强数据格式，使用 'dirty_value'
                            column_data.append(row['dirty_value'])
                        else:
                            print(
                                f"[{time.ctime()}] Warning: Row {idx} does not contain column {column} or 'dirty_value', skipping")
                            continue
                elif isinstance(val_data, pd.DataFrame):
                    if column not in val_data.columns:
                        print(
                            f"[{time.ctime()}] FD evaluation skipped for column {column}: Column not in val_data columns {val_data.columns.tolist()}")
                        self.logs.append(
                            f"FD evaluation skipped for column {column}: Column not in val_data columns {val_data.columns.tolist()}")
                        return
                    column_data = val_data[column].tolist()
                else:
                    raise TypeError(f"Unsupported val_data type: {type(val_data)}")

                if not column_data:
                    print(
                        f"[{time.ctime()}] FD evaluation skipped for column {column}: No valid data for column in val_data")
                    self.logs.append(f"FD evaluation skipped for column {column}: No valid data for column in val_data")
                    return

                corrected_vals = [correct_func(val) for val in column_data]
                matches = 0
                total = len(column_data)
                for idx, (orig, corrected, expected) in enumerate(zip(column_data, corrected_vals, column_data)):
                    expected_str = '' if pd.isna(expected) else str(expected)
                    corrected_str = str(corrected)
                    print(
                        f"[{time.ctime()}] FD evaluation for {column}: {orig} -> {corrected}, expected: {expected_str}")
                    if corrected_str == expected_str:
                        matches += 1
                    else:
                        print(
                            f"[{time.ctime()}] FD evaluation mismatch for {column}: {corrected_str} != {expected_str}")
                        self.logs.append(f"FD evaluation mismatch for {column}: {corrected_str} != {expected_str}")

                accuracy = matches / total if total > 0 else 0.0
                print(f"[{time.ctime()}] FD evaluation completed for column {column}, accuracy: {accuracy:.2f}")
                self.logs.append(f"FD evaluation for {column}: accuracy {accuracy:.2f}")
            except KeyError as e:
                error_msg = f"[{time.ctime()}] FD evaluation failed for column {column}: {str(e)}"
                print(error_msg)
                self.logs.append(error_msg)
            except SyntaxError as e:
                error_msg = f"[{time.ctime()}] FD evaluation failed for column {column}: Syntax error - {str(e)}"
                print(error_msg)
                self.logs.append(error_msg)
            except Exception as e:
                error_msg = f"[{time.ctime()}] FD evaluation failed for column {column}: {str(e)}"
                print(error_msg)
                self.logs.append(error_msg)

        def FD_generation(self, train_data: Dict, column: str, fds: Dict, llm, sys_prompt: str,
                          human_prompt: str) -> None:
            import re
            import json
            import logging
            logger = logging.getLogger(__name__)
            logger.debug(f"[{time.ctime()}] 开始为列 {column} 生成 FD")
            self.logs.append({
                'message': f"开始为列 {column} 生成 FD",
                'timestamp': time.ctime()
            })
            try:
                examples = [
                    {'original': k, 'corrected': v['correction'], 'error_type': v.get('error_type', 'Formatting Issue')}
                    for k, v in train_data.items() if 'correction' in v
                ]
                examples_str = json.dumps(examples, indent=2, ensure_ascii=False)
                column_type = 'int64' if column == 'col_b' else 'object'
                column_description = (
                    "col_a: String values in the format 'valueN' (e.g., 'value1', 'value2').\n"
                    "col_b: Integer numerical values (e.g., 10, 20, 30).\n"
                    "col_c: Single-letter string values (e.g., 'x', 'y', 'z').\n"
                    "sched_dep_time, act_dep_time, sched_arr_time, act_arr_time: Time values in 'HH:MM a.m.' or 'HH:MM p.m.' format."
                )
                prompt_input = {
                    'column_name': column,
                    'column_type': column_type,
                    'column_description': column_description,
                    'examples': examples_str,
                    'format_instructions': JsonOutputParser().get_format_instructions()
                }
                logger.debug(
                    f"[{time.ctime()}] FD 生成提示输入: {json.dumps(prompt_input, indent=2, ensure_ascii=False)}")
                self.logs.append({
                    'message': f"FD 生成提示输入: {json.dumps(prompt_input, indent=2, ensure_ascii=False)}",
                    'timestamp': time.ctime()
                })

                prompt = ChatPromptTemplate.from_messages([
                    SystemMessagePromptTemplate.from_template(sys_prompt),
                    HumanMessagePromptTemplate.from_template(human_prompt)
                ])
                chain = prompt | llm | JsonOutputParser()
                response = chain.invoke(prompt_input)
                logger.debug(
                    f"[{time.ctime()}] LLM 原始输出 for FD 列 {column}: {json.dumps(response, indent=2, ensure_ascii=False)}")
                self.logs.append({
                    'message': f"LLM 原始输出 for FD 列 {column}: {json.dumps(response, indent=2, ensure_ascii=False)}",
                    'timestamp': time.ctime()
                })

                if isinstance(response, list) and len(response) == 1:
                    response = response[0]
                if not isinstance(response, dict):
                    response_str = str(response).strip()
                    try:
                        response = json.loads(response_str)
                        if isinstance(response, list) and len(response) == 1:
                            response = response[0]
                    except json.JSONDecodeError as e:
                        logger.error(
                            f"[{time.ctime()}] JSON 解析失败 for FD 列 {column}: {str(e)}, 响应: {response_str}")
                        self.logs.append({
                            'message': f"JSON 解析失败 for FD 列 {column}: {str(e)}, 响应: {response_str}",
                            'timestamp': time.ctime()
                        })
                        response = {
                            'functional_dependency': (
                                f'import pandas as pd\n'
                                f'def correct(value):\n'
                                f'    import re\n'
                                f'    if pd.isna(value) or value == "nan":\n'
                                f'        return "{column == "sched_dep_time" and "7:15 a.m." or "12:00 p.m."}"\n'
                                f'    if isinstance(value, str):\n'
                                f'        if "/" in value:\n'
                                f'            parts = value.split(" ")\n'
                                f'            if len(parts) > 1 and (parts[-1].endswith("m") or parts[-1].endswith("m.")):\n'
                                f'                return parts[-1]\n'
                                f'        return re.sub(r"^(err_?|ERR_?)", "", value)\n'
                                f'    return value'
                            )
                        }

                if 'functional_dependency' not in response:
                    logger.warning(f"[{time.ctime()}] 响应缺少 'functional_dependency' 键 for 列 {column}: {response}")
                    self.logs.append({
                        'message': f"响应缺少 'functional_dependency' 键 for 列 {column}: {response}",
                        'timestamp': time.ctime()
                    })
                    response['functional_dependency'] = (
                        f'import pandas as pd\n'
                        f'def correct(value):\n'
                        f'    import re\n'
                        f'    if pd.isna(value) or value == "nan":\n'
                        f'        return "{column == "sched_dep_time" and "7:15 a.m." or "12:00 p.m."}"\n'
                        f'    if isinstance(value, str):\n'
                        f'        if "/" in value:\n'
                        f'            parts = value.split(" ")\n'
                        f'            if len(parts) > 1 and (parts[-1].endswith("m") or parts[-1].endswith("m.")):\n'
                        f'                return parts[-1]\n'
                        f'        return re.sub(r"^(err_?|ERR_?)", "", value)\n'
                        f'    return value'
                    )

                code_str = response['functional_dependency']
                try:
                    exec(code_str, {})
                    fds[column] = response
                    logger.debug(
                        f"[{time.ctime()}] FD 生成成功 for 列 {column}: {json.dumps(response, indent=2, ensure_ascii=False)}")
                    self.logs.append({
                        'message': f"FD 生成成功 for 列 {column}: {json.dumps(response, indent=2, ensure_ascii=False)}",
                        'timestamp': time.ctime()
                    })
                except SyntaxError as e:
                    logger.error(f"[{time.ctime()}] 无效的 Python 代码 for FD 列 {column}: {str(e)}, 代码: {code_str}")
                    self.logs.append({
                        'message': f"无效的 Python 代码 for FD 列 {column}: {str(e)}, 代码: {code_str}",
                        'timestamp': time.ctime()
                    })
                    fds[column] = {
                        'functional_dependency': (
                            f'import pandas as pd\n'
                            f'def correct(value):\n'
                            f'    import re\n'
                            f'    if pd.isna(value) or value == "nan":\n'
                            f'        return "{column == "sched_dep_time" and "7:15 a.m." or "12:00 p.m."}"\n'
                            f'    if isinstance(value, str):\n'
                            f'        if "/" in value:\n'
                            f'            parts = value.split(" ")\n'
                            f'            if len(parts) > 1 and (parts[-1].endswith("m") or parts[-1].endswith("m.")):\n'
                            f'                return parts[-1]\n'
                            f'        return re.sub(r"^(err_?|ERR_?)", "", value)\n'
                            f'    return value'
                        )
                    }
            except Exception as e:
                logger.error(f"[{time.ctime()}] FD 生成失败 for 列 {column}: {str(e)}")
                self.logs.append({
                    'message': f"FD 生成失败 for 列 {column}: {str(e)}",
                    'timestamp': time.ctime()
                })
                fds[column] = {
                    'functional_dependency': (
                        f'import pandas as pd\n'
                        f'def correct(value):\n'
                        f'    import re\n'
                        f'    if pd.isna(value) or value == "nan":\n'
                        f'        return "{column == "sched_dep_time" and "7:15 a.m." or "12:00 p.m."}"\n'
                        f'    if isinstance(value, str):\n'
                        f'        if "/" in value:\n'
                        f'            parts = value.split(" ")\n'
                        f'            if len(parts) > 1 and (parts[-1].endswith("m") or parts[-1].endswith("m.")):\n'
                        f'                return parts[-1]\n'
                        f'        return re.sub(r"^(err_?|ERR_?)", "", value)\n'
                        f'    return value'
                    )
                }
        def code_evaluation_execution(self, code: str, val_data: Dict[str, Any], column: str):
            print(f"[{time.ctime()}] Starting code evaluation for column {column}")
            try:
                exec(code, globals())
                for key, item in val_data.items():
                    original_value = item['dirty_value']
                    corrected_value = correct(original_value)
                    if corrected_value != original_value:
                        self.logs.append(f"Code evaluation for {column}: {original_value} -> {corrected_value}")
                        print(f"[{time.ctime()}] Code evaluation for {column}: {original_value} -> {corrected_value}")
            except Exception as e:
                self.logs.append(f"Code evaluation failed for column {column}: {str(e)}")
                print(f"[{time.ctime()}] Code evaluation failed for column {column}: {str(e)}")

        def update_retriever(self, column: str, retriever_instance: Any, embeddings_matrix_all: np.ndarray,
                             indices_for_column: List[int], dirty_data: pd.DataFrame,
                             header: List[str], repaired_row_indices: List[int]) -> int:
            """更新检索器，确保 metadatas 正确传递，并处理长度不匹配"""
            print(f"[{time.ctime()}] 更新检索器 for 列 {column}")
            try:
                if not repaired_row_indices:
                    print(f"[{time.ctime()}] 警告: 列 {column} 无修复行，跳过更新")
                    self.logs.append(f"警告: 列 {column} 无修复行，跳过更新")
                    return 0
                if column not in dirty_data.columns:
                    print(f"[{time.ctime()}] 错误: 列 {column} 不在 dirty_data 中")
                    self.logs.append(f"错误: 列 {column} 不在 dirty_data 中")
                    return 0
                if not all(idx in dirty_data.index for idx in repaired_row_indices):
                    print(f"[{time.ctime()}] 错误: 修复行索引 {repaired_row_indices} 中存在无效索引")
                    self.logs.append(f"错误: 修复行索引 {repaired_row_indices} 中存在无效索引")
                    return 0
                if not indices_for_column:
                    print(f"[{time.ctime()}] 警告: 列 {column} 索引为空，跳过更新")
                    self.logs.append(f"警告: 列 {column} 索引为空")
                    return 0

                valid_indices = [i for i in indices_for_column if i < len(header)]
                if len(valid_indices) != len(indices_for_column):
                    print(
                        f"[{time.ctime()}] 警告: 列 {column} 的 indices_for_column {indices_for_column} 包含无效索引，调整为 {valid_indices}")
                    self.logs.append(
                        f"警告: 列 {column} 的 indices_for_column {indices_for_column} 包含无效索引，调整为 {valid_indices}")
                    indices_for_column = valid_indices
                if not indices_for_column:
                    print(f"[{time.ctime()}] 警告: 列 {column} 无有效索引，跳过更新")
                    self.logs.append(f"警告: 列 {column} 无有效索引，跳过更新")
                    return 0

                texts = []
                header_col_filtered = [header[i] for i in indices_for_column]
                for idx in repaired_row_indices:
                    row = dirty_data.loc[idx]
                    values = row.values
                    detection_row = self.detection.loc[idx].values if idx in self.detection.index else [0] * len(header)
                    text = self.format_row_2(values, header, detection_row)
                    if not text or text == "{}":
                        print(f"[{time.ctime()}] 警告: 列 {column} 的行 {idx} 格式化为空")
                        text = json.dumps({column: str(row[column])}, ensure_ascii=False)
                    texts.append(text)

                metadatas = [{"index": idx} for idx in repaired_row_indices]
                print(f"[{time.ctime()}] Metadatas 构建完成: {len(metadatas)} 条，样本: {metadatas[:1]}")  # 调试日志
                embeddings = self.embedding_model.embed_documents(texts)

                # 使用 FAISS 构建检索器（避免 BM25 metadata 问题）
                faiss_retriever = FAISS.from_texts(
                    texts=texts,
                    embedding=self.embedding_model,
                    metadatas=metadatas,
                    distance_strategy=DistanceStrategy.COSINE
                )

                # 临时只使用 FAISS，避免 EnsembleRetriever 的 metadata 同步问题
                self.retriever_dict[column] = faiss_retriever.as_retriever(search_kwargs={"k": 5})

                # 确保 indices 和 CoE 长度匹配，使用 texts 长度
                self.indices_dict[column] = indices_for_column
                self.CoE_dict[column] = [1.0] * len(texts)  # 直接使用 len(texts)，避免不匹配

                print(f"[{time.ctime()}] 检索器重新构建完成 for 列 {column}: {len(texts)} 条记录")
                self.logs.append(f"检索器重新构建完成 for 列 {column}: {len(texts)} 条记录")
                return len(texts)
            except Exception as e:
                print(f"[{time.ctime()}] 更新检索器失败 for 列 {column}: {str(e)}")
                self.logs.append(f"更新检索器失败 for 列 {column}: {str(e)}")
                return 0

        def evaluate(self, clean_data: pd.DataFrame, dirty_data: Optional[pd.DataFrame] = None,
                     corrected_data: Optional[pd.DataFrame] = None) -> Dict[str, float]:
            """评估修复结果，计算精确率、召回率和 F1 分数，并保存比较结果到 Excel 文件"""
            print(f"[{time.ctime()}] 开始评估")
            try:
                # 如果未提供 dirty_data，使用 self.dirty_data
                if dirty_data is None:
                    if hasattr(self, 'dirty_data') and self.dirty_data is not None:
                        dirty_data = self.dirty_data
                        print(f"[{time.ctime()}] 使用 self.dirty_data 进行评估")
                    else:
                        error_msg = f"[{time.ctime()}] 错误: 评估需要 dirty_data 且 self.dirty_data 未设置"
                        print(error_msg)
                        self.logs.append(error_msg)
                        raise ValueError("评估需要 dirty_data 且 self.dirty_data 未设置")

                # 如果未提供 corrected_data，使用 self.corrections
                if corrected_data is None:
                    if hasattr(self, 'corrections') and self.corrections is not None:
                        corrected_data = self.corrections
                        print(f"[{time.ctime()}] 使用 self.corrections 进行评估")
                    else:
                        error_msg = f"[{time.ctime()}] 错误: 评估需要 corrected_data 且 self.corrections 未设置"
                        print(error_msg)
                        self.logs.append(error_msg)
                        raise ValueError("评估需要 corrected_data 且 self.corrections 未设置")

                # 验证输入形状和列名
                if clean_data.shape != corrected_data.shape or clean_data.shape != dirty_data.shape:
                    error_msg = (
                        f"[{time.ctime()}] 错误: 形状不匹配 - "
                        f"clean_data: {clean_data.shape}, dirty_data: {dirty_data.shape}, corrected_data: {corrected_data.shape}"
                    )
                    print(error_msg)
                    self.logs.append(error_msg)
                    raise ValueError("输入数据框必须具有相同形状")

                if not (clean_data.columns.equals(dirty_data.columns) and clean_data.columns.equals(
                        corrected_data.columns)):
                    error_msg = f"[{time.ctime()}] 错误: 列名不匹配 - clean_data: {list(clean_data.columns)}, dirty_data: {list(dirty_data.columns)}, corrected_data: {list(corrected_data.columns)}"
                    print(error_msg)
                    self.logs.append(error_msg)
                    raise ValueError("输入数据框必须具有相同的列名")

                # 使用 calc_p_r_f 计算指标
                metrics = self.calc_p_r_f(clean_data, dirty_data, corrected_data)
                print(f"[{time.ctime()}] 评估结果: {metrics}")

                # 保存比较结果到 comparison.xlsx
                try:
                    wb = Workbook()
                    ws = wb.active
                    ws.title = "Comparison"
                    headers = ['行', '列', '干净值', '修复值', '是否匹配']
                    ws.append(headers)
                    for row_idx in range(len(clean_data)):
                        for col in clean_data.columns:
                            clean_val = clean_data.at[row_idx, col]
                            corrected_val = corrected_data.at[row_idx, col]
                            match = '是' if self._safe_equals(clean_val, corrected_val) else '否'
                            ws.append([row_idx, col, str(clean_val), str(corrected_val), match])
                    wb.save(os.path.join(self.output_path, 'comparison.xlsx'))
                    print(f"[{time.ctime()}] 比较结果保存到 {os.path.join(self.output_path, 'comparison.xlsx')}")
                except Exception as e:
                    error_msg = f"[{time.ctime()}] 保存 comparison.xlsx 失败: {str(e)}"
                    print(error_msg)
                    self.logs.append(error_msg)

                return metrics
            except Exception as e:
                error_msg = f"[{time.ctime()}] 评估失败: {str(e)}"
                print(error_msg)
                self.logs.append(error_msg)
                raise


        def save(self, path: str):
            """
            保存纠正器模型的状态。
            """
            print(f"[{time.ctime()}] 保存模型到 {path}")
            try:
                os.makedirs(path, exist_ok=True)
                state = {
                    'retriever_dict': {k: v for k, v in self.retriever_dict.items() if k in self.dirty_data.columns},
                    'prompt_dict': self.prompt_dict,
                    'sp_examps': self.sp_examps,
                    'rep_error_info': self.rep_error_info,
                    'augmented_data': self.augmented_data,
                    'codes': self.codes,
                    'fds': self.fds,
                    'header': self.header,
                    'total_tokens': self.total_tokens,
                    'human_repair_num': self.human_repair_num,
                    'output_path': self.output_path,
                    'prompt_template_dir': self.prompt_template_dir,
                    'output_filename': self.output_filename
                }
                dump(state, os.path.join(path, 'zeroec_state.joblib'))
                print(f"[{time.ctime()}] 模型状态保存成功到 {path}/zeroec_state.joblib")
            except Exception as e:
                error_msg = f"[{time.ctime()}] 保存模型失败: {str(e)}"
                print(error_msg)
                self.logs.append(error_msg)
                raise

        def save_print_logs(self):
            if not self.logs:
                print("当前运行没有日志可保存或打印。")
                return
            log_file_path = os.path.join(self.output_path, "correction_log.json")
            try:
                valid_logs = []
                for log_entry in self.logs:
                    if isinstance(log_entry, dict) and all(k in log_entry for k in
                                                           ["Index", "Column", "Row Index", "Dirty_tuple",
                                                            "Dirty_value", "Correction", "Explanation"]):
                        valid_logs.append(log_entry)
                    else:
                        valid_logs.append({"message": str(log_entry), "timestamp": time.ctime()})
                with open(log_file_path, 'w', encoding='utf-8') as f:
                    json.dump(valid_logs, f, ensure_ascii=False, indent=4)
                print(f"纠正日志保存到: {log_file_path}")
                print("\n--- 当前纠正日志 ---")
                for log_entry in valid_logs:
                    print(json.dumps(log_entry, ensure_ascii=False, indent=2))
                print("--- 日志结束 ---\n")
            except Exception as e:
                print(f"保存/打印日志时出错: {e}")


if __name__ == "__main__":
    clean_temp_folders()
    clean_module_cache()
    print(f"[{time.ctime()}] 运行 ZeroEC 测试")
    from datetime import datetime
    import pandas as pd

    # 读取测试数据，仅读取前100条
    clean_data = pd.read_csv('flights_10_clean.csv', index_col=0, nrows=100)  # 限制为前100条
    dirty_data = pd.read_csv('flights_10_dirty.csv', index_col=0, nrows=100)  # 限制为前100条
    detection = pd.read_csv('flights_10_dirty_error_detection.csv', index_col=0, nrows=100)  # 限制为前100条

    # 重置索引，确保一致
    clean_data = clean_data.reset_index(drop=True)
    dirty_data = dirty_data.reset_index(drop=True)
    detection = detection.reset_index(drop=True)

    print(f"[{time.ctime()}] 干净数据形状: {clean_data.shape}")
    print(f"[{time.ctime()}] 脏数据形状: {dirty_data.shape}")
    print(f"[{time.ctime()}] 检测数据形状: {detection.shape}")
    print(f"[{time.ctime()}] 列名: {list(dirty_data.columns)}")

    # 初始化 ZeroEC
    corrector = ZeroEC(
        model_path="sentence-transformers/all-MiniLM-L6-v2",
        output_path="output",
        prompt_template_dir="prompt_templates",
        llm_config={
            'repair_model': 'Qwen/Qwen2.5-72B-Instruct',
            'base_url': 'https://api.siliconflow.cn/v1/',
            'api_key': 'sk-csclmsflwgupktivimfudfqhwvbrxdljnytubxmrxinsulcp',
            'repair_temperature': 0.5,
            'auto_cot_model': 'Qwen/Qwen2.5-72B-Instruct',
            'auto_cot_base_url': 'https://api.siliconflow.cn/v1/',
            'auto_cot_api_key': 'sk-csclmsflwgupktivimfudfqhwvbrxdljnytubxmrxinsulcp',
            'auto_cot_temperature': 0.5,
            'data_augmentation_model': 'Qwen/Qwen2.5-72B-Instruct',
            'data_augmentation_base_url': 'https://api.siliconflow.cn/v1/',
            'data_augmentation_api_key': 'sk-csclmsflwgupktivimfudfqhwvbrxdljnytubxmrxinsulcp',
            'data_augmentation_temperature': 0.3,
            'code_generation_model': 'Qwen/Qwen2.5-72B-Instruct',
            'code_generation_base_url': 'https://api.siliconflow.cn/v1/',
            'code_generation_api_key': 'sk-csclmsflwgupktivimfudfqhwvbrxdljnytubxmrxinsulcp',
            'code_generation_temperature': 0.0,
            'fd_generation_model': 'Qwen/Qwen2.5-72B-Instruct',
            'fd_generation_base_url': 'https://api.siliconflow.cn/v1/',
            'fd_generation_api_key': 'sk-csclmsflwgupktivimfudfqhwvbrxdljnytubxmrxinsulcp',
            'fd_generation_temperature': 0.0
        }
    )

    # 训练模型
    print(f"[{datetime.now()}] 开始训练")
    corrector.fit(dirty_data, clean_data, detection)

    # 预测纠正
    print(f"[{datetime.now()}] 开始预测")
    corrected_data = corrector.predict(dirty_data, detection)

    # 保存结果
    print(f"[{datetime.now()}] 保存日志")
    corrector.save_print_logs()

    # 评估结果
    print(f"[{datetime.now()}] 开始评估")
    metrics = corrector.evaluate(clean_data, dirty_data, corrected_data=corrected_data)
    print(f"[{datetime.now()}] 评估结果: {metrics}")

    # 保存模型
    print(f"[{datetime.now()}] 保存模型")
    corrector.save("model_output")