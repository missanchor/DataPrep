if __name__ == "__main__":
    try:
        clean_module_cache()

        llm_config = {
            'repair_model': 'Qwen/Qwen2.5-7B-Instruct',
            'auto_cot_model': 'Qwen/Qwen2.5-7B-Instruct',
            'data_augmentation_model': 'Qwen/Qwen2.5-7B-Instruct',
            'code_generation_model': 'Qwen/Qwen2.5-7B-Instruct',
            'fd_generation_model': 'Qwen/Qwen2.5-7B-Instruct',
            'api_key': 'sk-dqkzjxzxpxseetenozupfyvprhakjxwwchapqtboyguxgvzt',
            'api_key_auto_cot': 'sk-dqkzjxzxpxseetenozupfyvprhakjxwwchapqtboyguxgvzt',
            'api_key_data_augmentation': 'sk-dqkzjxzxpxseetenozupfyvprhakjxwwchapqtboyguxgvzt',
            'api_key_code_generation': 'sk-dqkzjxzxpxseetenozupfyvprhakjxwwchapqtboyguxgvzt',
            'api_key_fd_generation': 'sk-dqkzjxzxpxseetenozupfyvprhakjxwwchapqtboyguxgvzt',
            'base_url': 'https://api.siliconflow.cn/v1/',
            'repair_temperature': 0.5,
            'auto_cot_temperature': 0.5,
            'data_augmentation_temperature': 0.3,
            'code_generation_temperature': 0.0,
            'fd_generation_temperature': 0.0
        }

        clean_data_path = 'datasets/rayyan/rayyan_clean.csv'
        dirty_data_path = 'datasets/rayyan/rayyan_dirty.csv'
        detection_path = 'datasets/rayyan/rayyan_dirty_error_detection.csv'
        output_path = get_folder_name('runs_rayyan')
        os.makedirs(output_path, exist_ok=True)

        for path in [clean_data_path, dirty_data_path, detection_path]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"数据集文件不存在: {path}")

        print(f"[{time.ctime()}] 加载数据集")
        clean_data = pd.read_csv(clean_data_path, dtype=str, encoding='utf-8').fillna('null')
        dirty_data = pd.read_csv(dirty_data_path, dtype=str, encoding='utf-8').fillna('null')
        detection = pd.read_csv(detection_path, dtype=int)

        if not (clean_data.shape == dirty_data.shape == detection.shape):
            raise ValueError(f"数据集形状不一致: clean_data {clean_data.shape}, "
                           f"dirty_data {dirty_data.shape}, detection {detection.shape}")

        print(f"[{time.ctime()}] 初始化 ZeroEC")
        corrector = ZeroEC(
            model_path="all-MiniLM-L6-v2",
            output_path=output_path,
            prompt_template_dir="prompt_templates",
            llm_config=llm_config,
            output_filename="comparison.xlsx"
        )

        print(f"[{time.ctime()}] 开始训练模型")
        corrector.fit(dirty_data=dirty_data, clean_data=clean_data, detection=detection)

        print(f"[{time.ctime()}] 开始预测")
        corrected_data = corrector.predict(dirty_data=dirty_data, detection=detection)

        print(f"[{time.ctime()}] 开始评估")
        evaluation_results = corrector.evaluate(clean_data=clean_data)
        print(f"[{time.ctime()}] 评估结果: {evaluation_results}")

        print(f"[{time.ctime()}] 保存模型")
        corrector.save(output_path)

        print(f"[{time.ctime()}] 保存并打印日志")
        corrector.save_print_logs()

        clean_temp_folders()

    except Exception as e:
        print(f"[{time.ctime()}] 程序执行失败: {e}")
        raise
    finally:
        # 清理临时文件夹并关闭时间记录文件
        try:
            clean_temp_folders()
            corrector.f_time_cost.close()
        except Exception as e:
            print(f"[{time.ctime()}] 清理或关闭文件时出错: {e}")
            clean_temp_folders()
            corrector.f_time_cost.close()
