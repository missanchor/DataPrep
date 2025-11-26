import os
import sys
import time
import pandas as pd
import numpy as np
import joblib
import shutil
from dataprep.correction import zeroec
from dataprep.correction.zeroec import ZeroEC, clean_module_cache
from dataprep.correction.base import BaseDataCorrector

# 设置环境变量以避免 MiniBatchKMeans 内存泄漏
os.environ["OMP_NUM_THREADS"] = "1"

# 设置项目目录
project_dir = r"C:\Users\33328\Desktop\shuyan"
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)
print(f"[{time.ctime()}] 更新 sys.path: {sys.path}")
print(f"[{time.ctime()}] 脚本开始执行")

# 清理模块缓存
clean_module_cache()
print(f"[{time.ctime()}] 已清理模块缓存 for dataprep and zeroec")

# 验证 ZeroEC 和 BaseDataCorrector
print(f"[{time.ctime()}] ZeroEC 模块路径: {os.path.abspath(zeroec.__file__)}")
print(f"[{time.ctime()}] ZeroEC 方法: {dir(ZeroEC)}")
print(f"[{time.ctime()}] BaseDataCorrector 方法: {dir(BaseDataCorrector)}")
print(f"[{time.ctime()}] ZeroEC 基类: {ZeroEC.__bases__}")

def create_test_data():
    clean_data = pd.DataFrame({
        'col_a': [str(i + 1) for i in range(100)],
        'col_b': [str(i + 101) for i in range(100)],
        'col_c': [chr(97 + (i % 26)) for i in range(100)],
        'col_d': [chr(97 + (i % 26)) for i in range(100)]
    }, dtype=str)

    new_dirty_data = pd.DataFrame({
        'col_a': [f"err{i + 1}" if i % 2 == 0 else str(i + 1) for i in range(100)],
        'col_b': [f"err{i + 101}" if i % 2 == 0 else str(i + 101) for i in range(100)],
        'col_c': [f"err_{chr(97 + (i % 26))}" if i % 2 == 0 else chr(97 + (i % 26)) for i in range(100)],
        'col_d': [f"err_{chr(97 + (i % 26))}" if i % 2 == 0 else chr(97 + (i % 26)) for i in range(100)]
    }, dtype=str)

    new_detection = pd.DataFrame({
        'col_a': [1 if i % 2 == 0 else 0 for i in range(100)],
        'col_b': [1 if i % 2 == 0 else 0 for i in range(100)],
        'col_c': [1 if i % 2 == 0 else 0 for i in range(100)],
        'col_d': [1 if i % 2 == 0 else 0 for i in range(100)]
    })

    return clean_data, new_dirty_data, new_detection

def main():
    clean_data, new_dirty_data, new_detection = create_test_data()
    model_path = "model_output"

    print(f"[{time.ctime()}] 开始 ZeroEC 初始化")
    try:
        corrector = ZeroEC(
            model_path="sentence-transformers/all-MiniLM-L6-v2",  # 修正为完整模型标识符
            output_path="output",
            prompt_template_dir="prompt_templates",
            llm_config={
                'api_key': '',  # 替换为有效的 SiliconFlow API 密钥
                'base_url': 'https://api.siliconflow.cn/v1/',
                'repair_model': 'Qwen/Qwen2.5-7B-Instruct',
                'repair_temperature': 0.5
            }
        )
        print(f"[{time.ctime()}] ZeroEC 初始化成功")
    except Exception as e:
        print(f"[{time.ctime()}] ZeroEC 初始化失败: {str(e)}")
        return

    print(f"[{time.ctime()}] 开始拟合")
    try:
        corrector.fit(new_dirty_data, clean_data, new_detection)
        print(f"[{time.ctime()}] 拟合完成")
    except Exception as e:
        print(f"[{time.ctime()}] 拟合失败: {str(e)}")
        return

    print(f"[{time.ctime()}] 开始预测")
    try:
        new_corrected_data = corrector.predict(new_dirty_data, new_detection)
        print(f"[{time.ctime()}] 修复数据:\n{new_corrected_data.head()}")
    except Exception as e:
        print(f"[{time.ctime()}] 预测失败: {str(e)}")
        return

    print(f"[{time.ctime()}] 保存日志")
    try:
        corrector.save_print_logs()
        print(f"[{time.ctime()}] 日志保存完成")
    except Exception as e:
        print(f"[{time.ctime()}] 保存日志失败: {str(e)}")
        return

    print(f"[{time.ctime()}] 开始评估")
    try:
        metrics = corrector.evaluate(clean_data, new_dirty_data, new_corrected_data)
        print(f"[{time.ctime()}] 评估结果: {metrics}")
    except Exception as e:
        print(f"[{time.ctime()}] 评估失败: {str(e)}")
        return

    print(f"[{time.ctime()}] 保存模型")
    try:
        corrector.save(model_path)
        print(f"[{time.ctime()}] 模型保存到 {model_path}")
    except Exception as e:
        print(f"[{time.ctime()}] 保存模型失败: {str(e)}")
        return

    print(f"[{time.ctime()}] 加载模型")
    max_retries = 3
    for attempt in range(max_retries):
        try:
            loaded_corrector = ZeroEC.load(model_path, llm_config={
                'api_key': 'your_siliconflow_api_key_here',  # 替换为有效的 SiliconFlow API 密钥
                'base_url': 'https://api.siliconflow.cn/v1/',
                'repair_model': 'Qwen/Qwen2.5-7B-Instruct',
                'repair_temperature': 0.5
            })
            print(f"[{time.ctime()}] 模型加载成功")
            break
        except Exception as e:
            print(f"[{time.ctime()}] 模型加载失败 (尝试 {attempt + 1}/{max_retries}): {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(3)
            else:
                print(f"[{time.ctime()}] 模型加载最终失败: {str(e)}")
                return

    print(f"[{time.ctime()}] 使用加载的模型预测")
    try:
        new_corrected_data = loaded_corrector.predict(new_dirty_data, new_detection)
        print(f"[{time.ctime()}] 使用加载模型的修复数据:\n{new_corrected_data.head()}")
    except Exception as e:
        print(f"[{time.ctime()}] 使用加载模型预测失败: {str(e)}")

    if os.path.exists(model_path):
        print(f"[{time.ctime()}] 模型文件存在: {model_path}")
    else:
        print(f"[{time.ctime()}] 模型文件缺失: {model_path}")

if __name__ == "__main__":
    main()