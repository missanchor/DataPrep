import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from pydantic import BaseModel, Field
import json
from typing import Union, List, Dict
import numpy as np
from sentence_transformers import SentenceTransformer
import os
import time
import pandas as pd
from typing import Any, Dict, List
from langchain_core.embeddings import Embeddings

class RepairOutput(BaseModel):
    correction: Any = Field(description="修复后的值")
    chain_of_thought_for_correction: str = Field(description="修复的思考链")

class MyEmbeddings(Embeddings):
    def __init__(self, model_path: str):  # 添加 model_path 参数
        """初始化嵌入模型"""
        try:
            self.model = SentenceTransformer(model_path)
            self.model_dimension = self.model.get_sentence_embedding_dimension()
            print(f"[{time.ctime()}] 加载嵌入模型: {model_path}")
        except Exception as e:
            print(f"[{time.ctime()}] 加载嵌入模型失败: {str(e)}")
            raise

    def embed_query(self, text: Any) -> List[float]:
        """嵌入单个查询文本"""
        try:
            text = str(text)  # 强制转换为字符串
            embedding = self.model.encode(text, show_progress_bar=False)
            if isinstance(embedding, np.ndarray):
                embedding = embedding.tolist()
            if not embedding or len(embedding) != self.model_dimension:
                raise ValueError(f"嵌入向量无效，长度为 {len(embedding)}")
            print(f"[{time.ctime()}] 查询嵌入生成成功，长度: {len(embedding)}")
            return embedding
        except Exception as e:
            print(f"[{time.ctime()}] 查询嵌入失败: {str(e)}")
            return [0.0] * self.model_dimension

    def embed_documents(self, texts: List[str]) -> np.ndarray:  # 修改返回类型
        """嵌入文档列表"""
        print(f"[{time.ctime()}] 嵌入文档，文本数量: {len(texts)}")
        try:
            texts = [str(t) if pd.notna(t) else "" for t in texts]
            embeddings = self.model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
            if not isinstance(embeddings, np.ndarray):
                embeddings = np.array(embeddings)
            print(f"[{time.ctime()}] 嵌入文档完成，生成 {len(embeddings)} 个嵌入向量")
            return embeddings  # 直接返回 numpy.ndarray
        except Exception as e:
            print(f"[{time.ctime()}] 嵌入文档失败: {str(e)}")
            return np.zeros((len(texts), self.model_dimension))

    def embed_query(self, text: str) -> np.ndarray:
        print(f"[{time.ctime()}] 嵌入查询: {text[:50]}...")
        try:
            embedding = self.model.encode(text, normalize_embeddings=True)
            print(f"[{time.ctime()}] 嵌入查询完成")
            return embedding
        except Exception as e:
            print(f"[{time.ctime()}] 嵌入查询失败: {str(e)}")
            raise


def load_prompts(*file_paths: str) -> tuple:
    print(f"[{time.ctime()}] 加载提示模板: {file_paths}")
    prompts = []
    for file_path in file_paths:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                prompt = f.read()
                print(f"[{time.ctime()}] 成功加载提示模板: {file_path}")
                prompts.append(prompt)
        except FileNotFoundError:
            print(f"[{time.ctime()}] 警告: 提示文件 {file_path} 未找到，使用默认提示")
            prompts.append("Correct the erroneous value to its most likely correct form.")
        except Exception as e:
            print(f"[{time.ctime()}] 加载提示文件 {file_path} 失败: {str(e)}")
            prompts.append("Correct the erroneous value to its most likely correct form.")
    return tuple(prompts)

def load_examples(file_path: str) -> str:
    print(f"[{time.ctime()}] 加载示例文件: {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            examples = f.read()
            print(f"[{time.ctime()}] 成功加载示例文件: {file_path}")
            return examples
    except FileNotFoundError:
        print(f"[{time.ctime()}] 警告: 示例文件 {file_path} 未找到，返回默认示例")
        return "Example: 'err123' -> '123'"
    except Exception as e:
        print(f"[{time.ctime()}] 加载示例文件 {file_path} 失败: {str(e)}")
        return "Example: 'err123' -> '123'"

def form_examples(examples: List[Dict]) -> str:
    print(f"[{time.ctime()}] 格式化示例，示例数量: {len(examples)}")
    try:
        formatted = json.dumps(examples, indent=2, ensure_ascii=False)
        print(f"[{time.ctime()}] 示例格式化完成")
        return formatted
    except Exception as e:
        print(f"[{time.ctime()}] 示例格式化失败: {str(e)}")
        return "Example: {'original': 'err123', 'corrected': '123'}"