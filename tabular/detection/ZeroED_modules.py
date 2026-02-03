import os
import re
import json
import time
import random
import logging
import shutil
import traceback
import ast
import collections
from typing import Any, Dict, List, Optional
from datetime import datetime
from collections import defaultdict, Counter
from itertools import combinations

import pandas as pd
import numpy as np
from tqdm import tqdm

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mutual_info_score, pairwise_distances
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import MinMaxScaler
from kneed import KneeLocator
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage


# =============================================================================
# Section 1: 基础工具类 (Logger, Helpers)
# =============================================================================

class Logger:
    def __init__(self, resp_path):
        os.makedirs(resp_path, exist_ok=True)
        # 使用时间戳避免 logger 冲突
        self.logger = logging.getLogger(f'ZeroED_Logger_{time.time()}')

        if not self.logger.handlers:
            self.logger.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

            # File Handler
            log_file = os.path.join(resp_path, 'log.txt')
            fh = logging.FileHandler(log_file, encoding='utf-8')
            fh.setFormatter(formatter)
            self.logger.addHandler(fh)

            # Console Handler
            ch = logging.StreamHandler()
            ch.setFormatter(formatter)
            self.logger.addHandler(ch)

    def info(self, msg): self.logger.info(msg)

    def warning(self, msg): self.logger.warning(msg)

    def error(self, msg): self.logger.error(msg)


def get_ans_from_llm(prompt: str, api_key: str, model_name: str, base_url: str = 'https://api.siliconflow.cn/v1/', api_use: bool = True):
    """LLM 调用封装 (LangChain 优化版)"""
    if not api_use or not api_key:
        return "LLM API usage is disabled."

    try:
        # 1. 初始化时直接配置重试次数
        llm = ChatOpenAI(
            model=model_name,
            temperature=0.5,
            api_key=api_key,
            base_url=base_url,
            max_retries=3  # LangChain 会自动处理重试和指数退避
        )

        messages = [
            SystemMessage(content="You are a world-class data quality expert."),
            HumanMessage(content=prompt)
        ]

        resp = llm.invoke(messages)
        return resp.content

    except Exception as e:
        print(f"LLM API Error: {e}")
        return ""


def extract_json_from_text(text: str):
    """从 LLM 返回的文本中提取 JSON"""
    try:
        if "```json" in text:
            matches = re.findall(r'```json(.*?)```', text, re.DOTALL)
            if matches: return json.loads(matches[0].strip())
        elif "```" in text:
            matches = re.findall(r'```(.*?)```', text, re.DOTALL)
            if matches: return json.loads(matches[0].strip())
        # 尝试直接解析
        return json.loads(text.strip())
    except:
        return None


def extract_python_code(text: str) -> List[str]:
    """从文本中提取 Python 代码块"""
    return re.findall(r'```python(.*?)```', text, re.DOTALL)


# =============================================================================
# Section 2: 核心特征工程 (Single Source of Truth)
# =============================================================================

def extract_single_feature_vector(val: Any) -> np.ndarray:
    """
    统一的 9 维特征提取逻辑。
    用于 Phase 2 (Clustering), Phase 9 (Training) 和 Predict。
    """
    # 统一将数据转为字符串
    if pd.isna(val) or val == '':
        value_str = 'nan'
    else:
        value_str = str(val)

    features = np.zeros(9)

    # Feature 1: Numerical value extraction
    try:
        num_str = re.sub(r'[^\d.-]', '', value_str)
        if num_str and num_str not in ['-', '.']:
            features[0] = float(num_str)
        else:
            features[0] = 0.0
    except:
        features[0] = 0.0

    # Feature 2-4: Lengths and counts
    features[1] = len(value_str)
    features[2] = len(re.findall(r'\d', value_str))
    features[3] = len(re.findall(r'[a-zA-Z]', value_str))

    # Feature 5: Special character count
    features[4] = features[1] - features[2] - features[3]

    # Feature 6-7: ASCII of first and last char
    is_valid = value_str != 'nan' and value_str != ''
    features[5] = ord(value_str[0]) if is_valid else 0
    features[6] = ord(value_str[-1]) if is_valid else 0

    # Feature 8: Hash value (normalized)
    features[7] = (abs(hash(value_str)) % 10000 / 10000.0) if value_str != 'nan' else 0.0

    # Feature 9: Is Missing indicator
    features[8] = 1.0 if value_str == 'nan' else 0.0

    return features


def batch_extract_features(series: pd.Series) -> np.ndarray:
    """批量提取特征，返回 (N_samples, 9)"""
    return np.array([extract_single_feature_vector(v) for v in series])


# =============================================================================
# Section 3: Pipeline
# =============================================================================

def run_phase_1_correlation(df: pd.DataFrame, params: Dict, logger) -> Dict:
    """Phase 1: 计算相关属性 (NMI)"""
    if not params['related_attrs']:
        return {col: {} for col in df.columns}

    logger.info("Phase 1: Calculating Related Attributes (NMI)...")

    # 内部函数：计算两列 NMI
    def cal_nmi(c1, c2):
        mask = (c1 != 'nan') & (c2 != 'nan')
        c1, c2 = c1[mask], c2[mask]
        if len(c1) == 0: return 0.0

        def entropy(c):
            _, counts = np.unique(c, return_counts=True)
            probs = counts / len(c)
            return -sum(p * np.log2(p) for p in probs if p > 0)

        mi = mutual_info_score(c1, c2)
        e1, e2 = entropy(c1), entropy(c2)
        return 2 * mi / (e1 + e2) if (e1 + e2) > 0 else 0.0

    # 将所有列转为字符串处理 NMI
    df_str = df.astype(str)
    nmi_results = {}
    for c1, c2 in combinations(df.columns, 2):
        val = cal_nmi(df_str[c1], df_str[c2])
        nmi_results[(c1, c2)] = val

    # 筛选 Top K
    results = defaultdict(dict)
    for (c1, c2), val in nmi_results.items():
        results[c1][c2] = val
        results[c2][c1] = val

    top_results = {}
    for col, rels in results.items():
        sorted_rels = dict(sorted(rels.items(), key=lambda x: x[1], reverse=True)[:params['rel_top']])
        top_results[col] = sorted_rels

    return top_results


def train_single_column_pipeline(attr_name, df, related_attrs_map, params, logger):
    """
    针对单列的完整训练流程 (Phase 2 -> Phase 9)
    """

    # --- Phase 2: Clustering ---
    # 使用统一特征提取
    features = batch_extract_features(df[attr_name])
    if len(features) == 0:
        logger.warning(f"  Empty features for {attr_name}, skipping.")
        return None, []

    # 执行聚类
    centers, cluster_lists, val_feat_dict = _run_clustering_logic(features, df[attr_name], params, logger)
    logger.info(f"  Phase 2: Clustered into {len(centers)} representative centers.")

    # --- Phase 3: Distribution Analysis ---
    analy_res = ""
    if params['distri_analysis']:
        logger.info("  Phase 3: Running Distribution Analysis...")
        # 实例化分析器并运行
        analyzer = LLMDataDistrAnalyzer(df)
        prompt, _ = analyzer.generate_llm_prompt(attr_name)
        llm_resp = get_ans_from_llm(prompt, params['api_key'], params['model_name'], params['base_url'],params['api_use'])
        analy_res = analyzer.analyze_data(attr_name, llm_resp,
                                          os.path.join(params['result_dir'], f'{attr_name}_dist.txt'))

    # --- Phase 4: Guide Generation ---
    if params['guide_use']:
        logger.info("  Phase 4: Generating Guide...")
        prompt = kb_gen_prompt(attr_name, "dataset", centers, df, analy_res)
        get_ans_from_llm(prompt, params['api_key'], params['model_name'], params['base_url'],params['api_use'])

    # --- Phase 5: Labeling & Propagation ---
    logger.info("  Phase 5: Labeling and Propagating...")
    full_labels = _label_and_propagate(df, attr_name, centers, cluster_lists, params)

    clean_idxs = [i for i, l in full_labels.items() if l == 0]
    dirty_idxs = [i for i, l in full_labels.items() if l == 1]
    logger.info(f"  Labeled: {len(clean_idxs)} clean, {len(dirty_idxs)} dirty.")

    # --- Phase 6: Error Generation ---
    synthetic_feats = []
    if params['err_gen_use'] and clean_idxs:
        logger.info("  Phase 6: Generating Synthetic Errors...")
        synthetic_feats = _generate_synthetic_errors(df, attr_name, clean_idxs, dirty_idxs, params)
        logger.info(f"  Generated {len(synthetic_feats)} synthetic error samples.")

    # --- Phase 7 & 8: Function Generation & Filtering ---
    logger.info("  Phase 7 & 8: Function Generation & Filtering...")
    valid_funcs = _generate_and_filter_funcs(df, attr_name, clean_idxs, dirty_idxs, full_labels, params)
    logger.info(f"  Kept {len(valid_funcs)} valid functions.")

    # --- Phase 9: Local Model Training ---
    model = None
    if params['local_model_use']:
        logger.info("  Phase 9: Training Local Random Forest...")
        model = _train_local_model(df, attr_name, full_labels, val_feat_dict, synthetic_feats)
        if model:
            logger.info("  Local model trained successfully.")
        else:
            logger.warning("  Skipped local model (not enough class variety).")

    return model, valid_funcs


def predict_pipeline(df, models, funcs_map, local_model_use = True):
    """全局预测"""
    result_mask = pd.DataFrame(False, index=df.index, columns=df.columns)

    for col in df.columns:
        # 路径 A: 使用本地模型
        if local_model_use and col in models:
            clf = models[col]
            features = batch_extract_features(df[col])
            if len(features) > 0:
                preds = clf.predict(features)  # 1=Dirty, 0=Clean
                result_mask[col] = [bool(p) for p in preds]

        # 路径 B: 使用 Python 函数
        elif col in funcs_map:
            col_res = []
            funcs = funcs_map[col]
            for idx, row in df.iterrows():
                is_dirty = False
                for func_code in funcs:
                    # 如果任何一个函数返回 False (Dirty)，则认为该值为 Dirty
                    if _execute_cleaning_func(func_code, row, col) is False:
                        is_dirty = True
                        break
                col_res.append(is_dirty)
            result_mask[col] = col_res

    return result_mask


# =============================================================================
# Section 4: 内部实现逻辑 (Clustering, LLM Prompts, Helpers)
# =============================================================================

def _run_clustering_logic(feat, series, params, logger):
    """具体的聚类算法实现"""
    # 归一化
    if feat.shape[0] == 0:
        return [], [], defaultdict(list)

    feat_scaled = MinMaxScaler().fit_transform(feat)

    # 确定聚类数 (KneeLocator)
    n_clusters = 1
    max_k = min(feat.shape[0], 10)
    if max_k > 1:
        distortions = []
        for i in range(1, max_k + 1):
            km = KMeans(n_clusters=i, random_state=42, n_init=10).fit(feat_scaled)
            distortions.append(km.inertia_)
        try:
            kl = KneeLocator(range(1, max_k + 1), distortions, S=1.0, curve="convex", direction="decreasing")
            n_clusters = kl.elbow if kl.elbow else 1
        except:
            n_clusters = 1
    n_clusters = max(1, min(n_clusters, feat.shape[0]))

    # 执行聚类
    labels = np.zeros(feat.shape[0])
    original_indices = list(range(len(series)))
    centers = []
    cluster_lists = []

    if params['n_method'] == 'dbscan':
        # DBSCAN
        db = DBSCAN(eps=0.5, min_samples=2).fit(feat_scaled)
        labels = db.labels_
        unique_labels = set(labels)

        for k in unique_labels:
            if k == -1: continue  # Noise
            members = [original_indices[i] for i, l in enumerate(labels) if l == k]
            cluster_lists.append(members)
            # Find center (closest to mean)
            member_feats = feat_scaled[labels == k]
            centroid = np.mean(member_feats, axis=0)
            dists = pairwise_distances(member_feats, centroid.reshape(1, -1))
            center_idx_in_cluster = np.argmin(dists)
            centers.append(members[center_idx_in_cluster])

    else:
        # Agglomerative (Default)
        agg = AgglomerativeClustering(n_clusters=n_clusters).fit(feat_scaled)
        labels = agg.labels_

        for k in range(n_clusters):
            members = [original_indices[i] for i, l in enumerate(labels) if l == k]
            if not members: continue
            cluster_lists.append(members)

            member_feats = feat_scaled[labels == k]
            if len(member_feats) == 0: continue

            # Find center
            dists = pairwise_distances(member_feats)
            sum_dists = np.sum(dists, axis=1)
            center_idx_in_cluster = np.argmin(sum_dists)
            centers.append(members[center_idx_in_cluster])

    # 构建 Value -> Feature 映射
    val_feat_dict = defaultdict(list)
    for i, val in enumerate(series):
        val_feat_dict[str(val)].append(feat[i])  # 保存原始特征用于训练

    return centers, cluster_lists, val_feat_dict


def _label_and_propagate(df, attr, centers, clusters, params):
    """Phase 5: 询问 LLM 并传播标签"""
    labels = {}
    for c_idx in centers:
        row_str = str(df.iloc[c_idx].to_dict())
        prompt = error_check_prompt(row_str, attr)
        resp = get_ans_from_llm(prompt, params['api_key'], params['model_name'], params['base_url'],params['api_use'])

        # 解析 LLM 响应，寻找 "has_error": true
        if '"has_error' in resp and 'true' in resp.lower():
            labels[c_idx] = 1  # Dirty
        else:
            labels[c_idx] = 0  # Clean

    full_labels = {}
    for c_idx, label in labels.items():
        # 找到 c_idx 所在的簇，给簇内所有点打标
        for cluster in clusters:
            if c_idx in cluster:
                for member in cluster:
                    full_labels[member] = label
                break
    return full_labels


def _generate_synthetic_errors(df, attr, clean_idxs, dirty_idxs, params):
    """Phase 6: 生成合成错误"""
    clean_vals = df.iloc[clean_idxs[:5]].to_dict('records')
    dirty_vals = df.iloc[dirty_idxs[:5]].to_dict('records') if dirty_idxs else []

    prompt = create_err_gen_inst_prompt(clean_vals, dirty_vals, attr)
    resp = get_ans_from_llm(prompt, params['api_key'], params['model_name'], params['base_url'],params['api_use'])

    gen_list = extract_json_from_text(resp)
    new_feats = []

    if isinstance(gen_list, list):
        for item in gen_list:
            if isinstance(item, list) and len(item) >= 2:
                err_val = str(item[1])
                feat = extract_single_feature_vector(err_val)
                new_feats.append(feat)
    return new_feats


def _generate_and_filter_funcs(df, attr, clean_idxs, dirty_idxs, labels, params):
    """Phase 7 & 8"""
    clean_info = str(df.iloc[clean_idxs[:3]][attr].tolist()) if clean_idxs else "[]"
    err_info = str(df.iloc[dirty_idxs[:3]][attr].tolist()) if dirty_idxs else "[]"

    prompt = err_clean_func_prompt(attr, clean_info, err_info)
    resp = get_ans_from_llm(prompt, params['api_key'], params['model_name'], params['base_url'],params['api_use'])
    raw_funcs = extract_python_code(resp)

    valid = []
    for code in raw_funcs:
        # 在已有标签的数据上评估
        correct = 0
        total = 0
        for idx, lbl in labels.items():
            row = df.iloc[idx]
            # execute 返回 True 表示 Clean (label 0), False 表示 Dirty (label 1)
            is_clean_pred = _execute_cleaning_func(code, row, attr)
            is_clean_true = (lbl == 0)

            if is_clean_pred == is_clean_true:
                correct += 1
            total += 1

        acc = correct / total if total > 0 else 0
        if acc >= params['func_val_threshold']:
            valid.append(code)
    return valid


def _train_local_model(df, attr, labels, val_feat_dict, synth_feats):
    X, y = [], []
    # 真实数据
    for idx, lbl in labels.items():
        val_str = str(df.iloc[idx][attr])
        if val_str in val_feat_dict:
            X.append(val_feat_dict[val_str][0])  # 取第一个特征向量
            y.append(lbl)

    # 合成数据 (全是 Dirty)
    for f in synth_feats:
        X.append(f)
        y.append(1)

    if len(set(y)) > 1:
        clf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=1)
        clf.fit(X, y)
        return clf
    return None


def _execute_cleaning_func(code, row, attr):
    """安全执行生成的 Python 代码"""
    try:
        local_scope = {'row': row.to_dict(), 'attr': attr, 'pd': pd, 'np': np, 're': re}
        exec(code, {}, local_scope)
        for k, v in local_scope.items():
            if k.startswith('is_clean_') and callable(v):
                return v(row.to_dict(), attr)
    except:
        pass
    return True  # 默认 Clean


# =============================================================================
# Section 5: Prompts (Original Logic)
# =============================================================================

def kb_gen_prompt(attr_name, dataset_name, idx_list, dirty_csv, attr_analy_content):
    prompt = f'You are a top data scientist. Generate a guide for identifying errors in \'{attr_name}\' of \'{dataset_name}\':'
    if len(attr_analy_content) > 0:
        prompt += f"\n\nDistribution Analysis:\n{attr_analy_content}"

    # Examples
    examples = []
    for idx in idx_list[:20]:
        examples.append(str(dirty_csv.loc[int(idx), :].to_dict()))
    prompt += f'\n\nExamples:\n' + '\n'.join(examples)

    prompt += f'\n\nExplain \'{attr_name}\' and provide detection methods for: Pattern Violations, Missing Values, Constraints, Outliers, Typos, Common Knowledge Violations.'
    return prompt

def error_check_prompt(col_values, col_name):
    lines = col_values.strip().split('\n')
    try:
        col_list = re.findall(r'"([^"]+)"\s*:', lines[0])
    except json.JSONDecodeError as e:
        print(f"JSON Decode Error: {e}")
        print(f"Problematic JSON string: {lines[0]}")

    template_dict_1 = {key: f'{key}_example_val_1' for key in col_list}
    template_dict_2 = {key: f'{key}_example_val_2' for key in col_list}

    prompt = ""
    prompt += f"As a data quality expert, please first analyze attribute relations and analyze the '{col_name}' attribute values for potential errors. Ignore case sensitivity\n"
    prompt += f"Provide your analysis on `{col_name}` values in JSON format as follows, **do not care problems in other attributes**:\n\n"
    prompt += '''
```json
{'''
    prompt += f'''"column_name": "{col_name}",'''
    prompt += '''
  "entries": [
    {'''
    prompt += f'''\n"value_row": "{template_dict_1}",'''
    prompt += f'''\n"error_analysis": "[Brief explanation of the error analysis, if applicable]",'''
    prompt += f'''\n"has_error_in_{col_name}_value": true/false,'''
    prompt += '''
    },
    {'''
    prompt += f'''\n"value_row": "{template_dict_2}",'''
    prompt += f'''\n"error_analysis": "[Brief explanation of the error analysis, if applicable]",'''
    prompt += f'''\n"has_error_in_{col_name}_value": true/false,'''
    prompt += '''
    }
  ]
}
```
\n\n'''
    prompt += "If unsure, do not indicate an error.\n"
    prompt += "- Please ignore the case sensitivity issues.\n\n"
    prompt += "-----------------------------------------------\n\n"
    prompt += "Here are the given inputs:\n"
    prompt += f"Values of column '{col_name}' along with related attribute values:\n"
    prompt += f"'{col_values}'\n"
    prompt += f"Provide your analysis on `{col_name}` values in the required JSON format, **do not care problems in other attributes**:\n"
    return prompt

def create_err_gen_inst_prompt(clean_vals, dirty_vals, target_attribute, num_errors=20):
    if len(clean_vals) > 0:
        temp_vals = clean_vals[0]
    elif len(dirty_vals) > 0:
        temp_vals = dirty_vals[0]
    else:
        print(f"No vals in clean_vals and dirty_vals of attr {target_attribute}")
        temp_vals = f"{target_attribute}: none"
    attrs = re.findall(r"'(\w+)':", str(temp_vals))
    template_dict_1 = {key: f'{key}_val_1' for key in attrs}
    template_dict_1[target_attribute] = 'error_value_1'
    template_dict_2 = {key: f'{key}_val_2' for key in attrs}
    template_dict_2[target_attribute] = 'error_value_2'

    prompt = f"""
You are a data quality analyst with extensive experience in identifying and generating realistic data errors. Your task is to analyze a given dataset and generate plausible errors for a specific attribute, simulating real-world data quality issues.

I will provide you with a sample of **possible** clean and dirty values in a tabular format for various attributes.

Your objectives are to:
1. Analyze the data to identify patterns, relationships, and constraints between attributes.
2. Focus on the attribute named `{target_attribute}` and generate realistic errors that could occur in real-world scenarios.
3. Ensure the errors you generate are diverse and cover multiple error types.

Your final output **must be a single JSON array**.
The array should contain exactly {num_errors} items.
Each item in the array must be a list with four elements:
1. The target attribute name as a string (e.g., '{target_attribute}').
2. The generated error value as a string.
3. A reason for the error, as a string starting with 'Reason: '.
4. A dictionary representing the complete data row with the error injected.

Do not include any other text, explanation, or conversational phrases in your output.
Do not be the same as the reference values.

The types of errors to consider include:
1. Pattern Violations: Values that don't match the expected format.
2. Explicit/Implicit Missing Values: Null values or placeholders for missing data.
3. Constraints Violations: Values that conflict with other columns or violate business rules.
4. Out-of-domain values: Values outside the expected range or set.
5. Typos: Spelling or data entry errors.
6. Violate common knowledge: Values that contradict widely known facts.
"""
    prompt += f"For the attribute `{target_attribute}`, here are the given **possible** clean tuples:\n"
    prompt += '\n'.join([str(i) for i in clean_vals]) + '\n'
    prompt += f"There are also some **possible** wrong tuples for reference:\n"
    prompt += '\n'.join([str(i) for i in dirty_vals]) + '\n\n'
    prompt += f"Please analyze the error pattern and generate {num_errors} realistic errors specifically for the attribute `{target_attribute}`. Return the final JSON array only."

    return prompt

def pre_func_prompt(attr_name, data_example):
    prompt = (
        f"You are a Data Cleaning Specialist tasked with distinguishing between clean and dirty cells in the `{attr_name}`.\n\n"

        f"Here are examples for the '{attr_name}' column:\n"
        f"{data_example}\n\n"

        "Your task:\n"
        f"1. Analyze the `{attr_name}` column values.\n"
        "2. Create precise judge functions in Python that:\n"
        f"- Receive the row content of the `{attr_name}` column\n"
        "- Return True for clean values, False for dirty values\n"
        "- Use the naming convention 'is_clean_[judgment]'\n"
        "- Cover different perspectives of cleanliness as possible\n"
        "- Do not contain blank lines inner functions\n\n"

        "Example function code snippet:\n"
        "```python "
        f"def is_clean_[judgment](row, attr):\n"
        f"    # Value of `{attr_name}` is row[attr]\n"
        "    # Your logic here\n"
        "    return True  # or False\n"
        "```\n"
        "Provide your functions below:\n"
    )
    return prompt

def err_clean_func_prompt(attr_name, clean_info, errs_info):
    prompt = (
        f"You are a Data Cleaning Specialist tasked with identifying and distinguishing between clean and dirty cells in the `{attr_name}` column.\n\n"
        f"Clean examples for the '{attr_name}' column:\n"
        f"{clean_info}\n\n"
        f"Error examples for the '{attr_name}' column:\n"
        f"{errs_info}\n\n"

        "Your task:\n"
        f"1. Analyze the `{attr_name}` column values.\n"
        "2. Compare the differences between clean and dirty values.\n"
        "3. Create precise judge functions in Python that:\n"
        f"- Receive the row content of the `{attr_name}` column\n"
        "- Return True for clean values, False for dirty values\n"
        "- Use the naming convention 'is_clean_[judgment]'\n"
        "- Cover different perspectives of cleanliness as possible\n"
        "- Do not contain blank lines inner functions\n\n"

        "Example function code snippet:\n"
        "```python "
        f"def is_clean_[judgment](row, attr):\n"
        f"    # Value of `{attr_name}` is row[attr]\n"
        "    # Your logic here\n"
        "    return True  # or False\n"
        "```\n"
        "Provide your functions below:\n"
    )
    return prompt

# =============================================================================
# Section 6: LLM Analyzer Class (Copied & Cleaned)
# =============================================================================

class LLMDataDistrAnalyzer:
    def __init__(self, dirty_csv):
        self.dirty_csv = dirty_csv

    def get_column_examples(self, attr_name: str, n_samples: int = 10) -> str:
        """Get sample values from the specified column."""
        if attr_name not in self.dirty_csv.columns:
            return f"Error: Column {attr_name} not found"

        samples = self.dirty_csv.sample(n=min(n_samples, len(self.dirty_csv)))
        result_str = '\n'.join(
            [', '.join([f"{attr}: {value}" for attr, value in row.items()]) for row in samples.to_dict('records')])
        return f"Examples of values in column '{attr_name}': \n{result_str}"

    def generate_llm_prompt(self, attr_name: str) -> str:
        """Generate prompt for LLM to create analysis functions."""
        examples = self.get_column_examples(attr_name)

        prompt = f"""
Based on the column '{attr_name}' with examples: {examples}

Please generate Python functions to analyze the data distribution from various perspectives, so that we can verify whether an error is reasonable or not. 
Each function should:
1. Take parameters (dirty_csv: dataframe, attr_name: str), regard all values in dirty_csv are **strings**
2. Return a string containing the **detailed** analysis results
3. Do not enumerate/count all values, showing representative ones
4. **Also import necessary libraries**

Example function code snippet:\n
```python 
def distr_analysis_[perspective](dirty_csv, attr_name):
    # Your logic here
    return 'Detailed description of the analysis results'
```\n
Provide your functions below:\n
        """
        return prompt, examples

    def validate_and_clean_function(self, function_code: str) -> str:
        try:
            ast.parse(function_code)
            return function_code
        except SyntaxError:
            return None

    def execute_function(self, function_code: str, attr_name: str) -> str:
        """Execute a single extracted function safely."""
        try:
            # Create namespace for function execution
            namespace = {
                'dirty_csv': self.dirty_csv,
                'attr_name': attr_name,
                'pd': pd,
                'np': np,
                'datetime': datetime,
                'Counter': Counter,
                're': re,
                'collections': collections,
            }

            setup_code = """
import pandas as pd
import numpy as np
import re
from datetime import datetime
from typing import Dict, List
from collections import Counter
            """
            exec(setup_code + "\n" + function_code, globals(), namespace)
            function_name = function_code.split("def ")[1].split("(")[0].strip()
            func = namespace[function_name]
            result = func(self.dirty_csv, attr_name)
            return result

        except Exception as e:
            return f"Error executing function: {str(e)}"

    def analyze_data(self, attr_name: str, llm_response: str, output_file: str) -> Dict:
        functions = extract_func(llm_response)
        self.functions = functions

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"Analysis Results for {attr_name}\n")
            f.write("=" * 50 + "\n")
        results = ""
        for i, func_code in enumerate(functions, 1):
            clean_code = self.validate_and_clean_function(func_code)
            if clean_code:
                result = self.execute_function(clean_code, attr_name)
                if result is not None:
                    results += f"\n\n=== Data Distribution Analysis {i} ===\n"
                    if isinstance(result, str) and len(result.split('\n')) > 30:
                        result = '\n'.join(result.split('\n')[:30]) + "\n... Too long, only sample some examples."
                    results = results + '\n' + result
                    with open(output_file, 'a', encoding='utf-8') as f:
                        f.write(f"\n\n=== Data Distribution Analysis Function {i} ===\n")
                        f.write(clean_code)
                        f.write("\n**Running Results:**\n")
                        f.write(str(result))

        return results


def extract_func(text_content):
    try:
        code_blocks = re.findall(r'```(.*?)```', text_content, re.DOTALL)
    except re.error as e:
        print(f"Regex error: {e}")
        return [], []
    func_list = []
    for code_block in code_blocks:
        functions = re.findall(r'def \w+\(.*?\):\n(?:[ \t]*\n)*(?: .*\n)+', code_block)
        for function in functions:
            try:
                function_name = re.findall(r'def (\w+)', function)[0]
            except IndexError:
                print("Function name not found in the function definition.")
                continue
            func_list.append(function)
    return func_list