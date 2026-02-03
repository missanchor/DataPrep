import os
import time
import pandas as pd
import numpy as np
from langchain_core.output_parsers import JsonOutputParser
from dataprep.base import BaseEstimator
import dataprep.tabular.correction.ZeroEC_modules as modules


class ZeroEC(BaseEstimator):
    def __init__(self,
                 model_name="qwen2.5-7b",
                 openai_api_base="http://localhost:8000/v1",
                 openai_api_key="EMPTY",
                 embedding_model_path='./all-MiniLM-L6-v2',
                 human_repair_num=10,
                 output_dir='runs_rayyan',
                 clean_data_path='datasets/rayyan/rayyan_clean.csv',
                 dirty_data_path='datasets/rayyan/rayyan_dirty.csv',
                 detection_path='datasets/rayyan/rayyan_dirty_error_detection.csv',
                 prompt_dir='prompt_templates',
                 max_workers=3,
                 **kwargs):
        # 1. 初始化环境
        output_path = modules.get_folder_name(output_dir)
        self.f_time_cost = open(os.path.join(output_path, 'rayyan_time_cost.txt'), 'a', encoding='utf-8')

        # 2. 准备全局状态字典
        self.params = {
            'MODEL_NAME': model_name,
            'OPENAI_API_BASE': openai_api_base,
            'OPENAI_API_KEY': openai_api_key,
            'EMBEDDING_MODEL_PATH': embedding_model_path,
            'human_repair_num': human_repair_num,
            'output_path': output_path,
            'max_workers': max_workers,
            'prompt_dir': prompt_dir,
            'clean_data_path': clean_data_path,
            'dirty_data_path': dirty_data_path,
            'detection_path': detection_path,
            'f_time_cost': self.f_time_cost,
            'logs': [],
            'total_tokens': 0,
            # 中间状态容器
            'retriever_dict': {},
            'indices_dict': {},
            'CoE_dict': {},
            'retrieved_tuples': {},
            'sp_examps': {},
            'rep_error_info': {},
            'codes': {},
            'fds': {},
            'prompt_dict': {},
            'train_data': {},
            'val_data': {}
        }

        # 4. 加载资源 (Prompt, Data, Models)
        self._initialize_resources()

    def _initialize_resources(self):
        """加载数据、Prompt和模型，存入 self.params"""
        # 加载 Prompts
        prompts = modules.load_all_prompts(self.params['prompt_dir'])
        self.params.update(prompts)

        # 加载数据
        data_dict = modules.load_datasets(self.params)
        self.params.update(data_dict)
        self.params['corrections'] = self.params['dirty_data'].copy()  # 初始化 corrections

        # 初始化 LLMs 和 Parser
        models = modules.init_all_models(self.params)
        self.params.update(models)
        self.params['parser'] = JsonOutputParser(pydantic_object=modules.Output)

    def train(self):
        """
        Train 阶段：
        1. Embedding & 候选选择 (Embedding & Selection)
        2. 模拟人工修复 (Human Repair Simulation)
        3. Auto-CoT 生成 (Auto-CoT Generation)
        4. 代码和函数依赖生成 (Code & FD Generation)
        """
        print(f"Total errors: {self.params['detection'].sum().sum()}")
        self.start_time = time.time()

        # Phase 1: Embedding & Selection
        modules.run_embedding_and_selection(self.params)

        # Phase 2: Human Repair Simulation
        modules.simulate_human_repair(self.params)

        # Phase 3: Auto-CoT Generation (Retriever + Few-shot)
        modules.run_auto_cot_generation(self.params)

        # Phase 4: Code & FD Generation
        modules.run_code_fd_generation(self.params)

    def predict(self):
        """
        Predict 阶段：
        1. 执行生成的代码/FD (Code/FD Execution)
        2. 更新检索器 (Retriever Update)
        3. 全局检索 (Retrieval)
        4. LLM 最终修复 (LLM Repair)
        5. 评估 (Evaluation)
        """
        # Phase 1: Code & FD Execution
        modules.run_code_fd_execution(self.params)

        # Phase 2: Retriever Update
        modules.run_retriever_update(self.params)

        # Phase 3: Retrieval
        modules.run_retrieval(self.params)

        # Phase 4: LLM Repair
        modules.run_llm_repair(self.params)

        # Phase 5: Evaluation
        execution_time = time.time() - self.start_time
        modules.run_evaluation(self.params, execution_time)

        # 清理资源
        self.f_time_cost.close()

        return self.params['corrections']

    def train_and_predict(self):
        self.train()
        return self.predict()


if __name__ == "__main__":
    zeroec = ZeroEC()
    zeroec.train()
    zeroec.predict()