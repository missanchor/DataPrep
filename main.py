import os
import sys
import time
import traceback
import numpy as np
import pandas as pd
import torch
import random
import asyncio
import contextlib
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, HTTPException, WebSocket, Query
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# ==================== 算法与评估库引入 ====================
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer, KNNImputer
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, precision_score, recall_score

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from dataprep.tabular.imputation.GAIN import GAIN
from dataprep.tabular.imputation.VAEGAIN import VAEGAIN
from dataprep.tabular.imputation.SCIS import SCIS
from dataprep.tabular.detection.ZeroED import ZeroED
from dataprep.tabular.correction.ZeroEC import ZeroEC

app = FastAPI(title="DataPrep Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def setup_seed(seed=49):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calc_rmse(data_true, data_imputed, mask):
    if isinstance(data_imputed, torch.Tensor):
        data_imputed = data_imputed.cpu().detach().numpy()
    elif isinstance(data_imputed, pd.DataFrame):
        data_imputed = data_imputed.values
    if isinstance(mask, pd.DataFrame): mask = mask.values
    if isinstance(data_true, pd.DataFrame): data_true = data_true.values

    missing_mask = (mask == 0)
    if np.sum(missing_mask) == 0: return 0.0
    return float(np.sqrt(np.mean((data_true[missing_mask] - data_imputed[missing_mask]) ** 2)))


def calc_mae(data_true, data_imputed, mask):
    if isinstance(data_imputed, torch.Tensor):
        data_imputed = data_imputed.cpu().detach().numpy()
    elif isinstance(data_imputed, pd.DataFrame):
        data_imputed = data_imputed.values
    if isinstance(mask, pd.DataFrame): mask = mask.values
    if isinstance(data_true, pd.DataFrame): data_true = data_true.values

    missing_mask = (mask == 0)
    if np.sum(missing_mask) == 0: return 0.0
    return float(np.mean(np.abs(data_true[missing_mask] - data_imputed[missing_mask])))


def load_and_prep_detection_data(path_dirty, path_gt):
    df_dirty = pd.read_csv(path_dirty, index_col=0) if 'index_col' in str(
        pd.read_csv(path_dirty, nrows=1).columns) else pd.read_csv(path_dirty)
    df_gt = pd.read_csv(path_gt)

    if 'index' in df_gt.columns: df_gt.drop(columns=['index'], inplace=True)
    if 'index' in df_dirty.columns: df_dirty.drop(columns=['index'], inplace=True)

    if df_gt.dtypes.iloc[0] == object:
        df_gt = df_gt.replace({'True': True, 'False': False, '1': True, '0': False})

    y_true = df_gt.values.any(axis=1).astype(int)

    df_sklearn = df_dirty.copy()
    imputer = SimpleImputer(strategy='most_frequent')
    df_sklearn = pd.DataFrame(imputer.fit_transform(df_sklearn), columns=df_sklearn.columns)

    for col in df_sklearn.columns:
        if df_sklearn[col].dtype == 'object':
            df_sklearn[col] = LabelEncoder().fit_transform(df_sklearn[col].astype(str))

    return df_dirty, df_sklearn, y_true


class SklearnCorrector:
    def __init__(self, strategy='most_frequent'):
        self.strategy = strategy
        self.encoders = {}

    def correct(self, df_dirty, df_mask):
        X = df_dirty.copy()
        X = X.mask(df_mask.astype(bool), np.nan)
        X_encoded = X.copy()
        for col in X.columns:
            if X[col].dtype == 'object' or pd.api.types.is_string_dtype(X[col]):
                le = LabelEncoder()
                valid_vals = X[col].dropna().unique()
                le.fit(valid_vals.astype(str))
                self.encoders[col] = le
                non_null_mask = X[col].notnull()
                X_encoded.loc[non_null_mask, col] = le.transform(X.loc[non_null_mask, col].astype(str))
                X_encoded[col] = pd.to_numeric(X_encoded[col], errors='coerce')

        imputer = SimpleImputer(strategy='most_frequent') if self.strategy == 'most_frequent' else KNNImputer(
            n_neighbors=5)
        X_imputed = imputer.fit_transform(X_encoded)
        df_imputed = pd.DataFrame(X_imputed, columns=X.columns, index=X.index)

        df_final = df_imputed.copy()
        for col, le in self.encoders.items():
            if col in df_imputed.columns:
                series_col = pd.to_numeric(df_imputed[col], errors='coerce')
                vals = series_col.round().astype(int).clip(0, len(le.classes_) - 1)
                df_final[col] = le.inverse_transform(vals)
        return df_final


def evaluate_correction_metrics(clean_data, dirty_data, corrected_data, df_mask):
    common_idx = clean_data.index.intersection(corrected_data.index).intersection(dirty_data.index)
    common_col = clean_data.columns.intersection(corrected_data.columns).intersection(dirty_data.columns)

    gt = clean_data.loc[common_idx, common_col]
    dirty = dirty_data.loc[common_idx, common_col]
    pred = corrected_data.loc[common_idx, common_col]
    mask = df_mask.loc[common_idx, common_col].astype(bool)

    def safe_str_format(df):
        return df.fillna("").astype(str).map(lambda x: x.strip())

    gt_str = safe_str_format(gt)
    dirty_str = safe_str_format(dirty)
    pred_str = safe_str_format(pred)
    mask_bool = mask.values

    actual_errors_num = mask_bool.sum()

    if actual_errors_num == 0:
        return 0.0, 0.0, 0.0

    is_correct = (pred_str == gt_str).values
    correctly_fixed_num = (is_correct & mask_bool).sum()

    is_modified = (pred_str != dirty_str).values
    total_modified_num = is_modified.sum()

    Recall = correctly_fixed_num / actual_errors_num if actual_errors_num > 0 else 0.0
    Precision = correctly_fixed_num / total_modified_num if total_modified_num > 0 else 0.0
    F1 = (2 * Precision * Recall) / (Precision + Recall) if (Precision + Recall) > 0 else 0.0

    return Precision, Recall, F1


class WsLogCatcher:
    def __init__(self, loop, queue, original_stream):
        self.loop = loop
        self.queue = queue
        self.original_stream = original_stream

    def write(self, s):
        self.original_stream.write(s)
        self.original_stream.flush()
        if s:
            asyncio.run_coroutine_threadsafe(self.queue.put({"log": s}), self.loop)
        return len(s)

    def flush(self):
        self.original_stream.flush()


def generate_result_data(df: pd.DataFrame, df_mask: pd.DataFrame, max_rows: int = 1000) -> dict:
    df_head = df.head(max_rows).copy()
    mask_head = df_mask.head(max_rows).copy()
    df_head = df_head.astype(object).where(pd.notnull(df_head), None)

    return {
        "columns": df_head.columns.tolist(),
        "rows": df_head.to_dict(orient="records"),
        "masks": mask_head.astype(bool).to_dict(orient="records")
    }


@app.get("/api/preview")
def preview_data(path: str = Query(..., description="CSV File Path")):
    try:
        if not os.path.exists(path):
            raise HTTPException(status_code=404, detail=f"找不到文件，请检查路径: {path}")

        df = pd.read_csv(path)
        preview_df = df.head(1000).replace({np.nan: None})

        stats = []
        for col in df.columns:
            col_data = df[col]
            missing_count = int(col_data.isnull().sum())

            if pd.api.types.is_numeric_dtype(col_data):
                stats.append({
                    "name": col, "type": "Numeric", "missing": missing_count,
                    "mean": round(float(col_data.mean()), 2) if not pd.isnull(col_data.mean()) else None,
                    "std": round(float(col_data.std()), 2) if not pd.isnull(col_data.std()) else None,
                    "min": round(float(col_data.min()), 2) if not pd.isnull(col_data.min()) else None,
                    "max": round(float(col_data.max()), 2) if not pd.isnull(col_data.max()) else None
                })
            else:
                stats.append({
                    "name": col, "type": "Categorical", "missing": missing_count,
                    "unique": int(col_data.nunique())
                })

        return {
            "columns": df.columns.tolist(),
            "rows": preview_df.to_dict(orient="records"),
            "stats": stats
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.websocket("/api/ws/run_task")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    payload = await websocket.receive_json()

    method = payload.get("method")
    paths = payload.get("paths", {})
    params = payload.get("params", {})
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    loop = asyncio.get_running_loop()
    log_queue = asyncio.Queue()

    def run_ml_task():
        setup_seed(49)
        stdout_catcher = WsLogCatcher(loop, log_queue, sys.stdout)
        stderr_catcher = WsLogCatcher(loop, log_queue, sys.stderr)

        with contextlib.redirect_stdout(stdout_catcher), contextlib.redirect_stderr(stderr_catcher):
            try:
                print(f"========== 开始执行任务: {method} ==========\n")
                result_data = None

                if method in ["GAIN", "VAEGAIN", "SCIS"]:
                    print("Loading data...")
                    df_missing = pd.read_csv(paths.get("dataPath"))
                    columns = df_missing.columns
                    data_missing = df_missing.values
                    data_true = pd.read_csv(paths.get("groundTruthPath")).values
                    mask = pd.read_csv(paths.get("missingMaskPath")).values

                    imp_bayes = IterativeImputer(estimator=BayesianRidge(), max_iter=10, random_state=42)
                    res_bayes = imp_bayes.fit_transform(data_missing)
                    rmse_bayes = calc_rmse(data_true, res_bayes, mask)
                    mae_bayes = calc_mae(data_true, res_bayes, mask)

                    imp_rf = IterativeImputer(
                        estimator=RandomForestRegressor(n_estimators=10, max_depth=10, n_jobs=-1, random_state=42),
                        max_iter=5, random_state=42)
                    res_rf = imp_rf.fit_transform(data_missing)
                    rmse_rf = calc_rmse(data_true, res_rf, mask)
                    mae_rf = calc_mae(data_true, res_rf, mask)

                    print(f"Running {method}...")
                    params['device'] = device
                    if method == "GAIN":
                        model = GAIN(**params)
                    elif method == "VAEGAIN":
                        model = VAEGAIN(**params)
                    elif method == "SCIS":
                        model = SCIS(**params)

                    res_ours = model.train_and_predict(data_missing, mask)
                    rmse_ours = calc_rmse(data_true, res_ours, mask)
                    mae_ours = calc_mae(data_true, res_ours, mask)

                    result = {
                        "rmse_bayes": round(rmse_bayes, 4), "mae_bayes": round(mae_bayes, 4),
                        "rmse_rf": round(rmse_rf, 4), "mae_rf": round(mae_rf, 4),
                        "rmse_ours": round(rmse_ours, 4), "mae_ours": round(mae_ours, 4)
                    }

                    df_res = pd.DataFrame(res_ours, columns=columns)
                    df_highlight = pd.DataFrame(mask == 0, columns=columns)
                    result_data = generate_result_data(df_res, df_highlight)

                elif method == "ZeroED":
                    df_raw, df_enc, y_true = load_and_prep_detection_data(paths.get("dataPath"),
                                                                          paths.get("errorDetectionPath"))
                    contamination_rate = max(0.01, min(0.5, np.sum(y_true) / len(y_true)))

                    iso = IsolationForest(contamination=contamination_rate, random_state=49, n_jobs=-1)
                    y_pred_iso = np.where(iso.fit_predict(df_enc.values) == -1, 1, 0)

                    lof = LocalOutlierFactor(contamination=contamination_rate, n_neighbors=20)
                    y_pred_lof = np.where(lof.fit_predict(df_enc.values) == -1, 1, 0)

                    print("Running ZeroED...")
                    detector = ZeroED(**params)
                    detector.train(df_raw)

                    df_pred_mask = detector.predict(df_raw)
                    if isinstance(df_pred_mask, np.ndarray):
                        df_pred_mask = pd.DataFrame(df_pred_mask, columns=df_raw.columns)

                    y_pred_ours = df_pred_mask.values.any(axis=1).astype(int)

                    result = {
                        "det_prec_iso": round(precision_score(y_true, y_pred_iso, zero_division=0), 4),
                        "det_rec_iso": round(recall_score(y_true, y_pred_iso, zero_division=0), 4),
                        "det_f1_iso": round(f1_score(y_true, y_pred_iso, zero_division=0), 4),
                        "det_prec_lof": round(precision_score(y_true, y_pred_lof, zero_division=0), 4),
                        "det_rec_lof": round(recall_score(y_true, y_pred_lof, zero_division=0), 4),
                        "det_f1_lof": round(f1_score(y_true, y_pred_lof, zero_division=0), 4),
                        "det_prec_ours": round(precision_score(y_true, y_pred_ours, zero_division=0), 4),
                        "det_rec_ours": round(recall_score(y_true, y_pred_ours, zero_division=0), 4),
                        "det_f1_ours": round(f1_score(y_true, y_pred_ours, zero_division=0), 4)
                    }
                    result_data = generate_result_data(df_raw, df_pred_mask.astype(bool))

                elif method == "ZeroEC":
                    df_clean = pd.read_csv(paths.get("cleanDataPath"), index_col=0)
                    df_dirty = pd.read_csv(paths.get("dataPath"), index_col=0)
                    df_mask = pd.read_csv(paths.get("detectionPath"))

                    if 'index' in df_mask.columns: df_mask.drop(columns=['index'], inplace=True)
                    if 'Unnamed: 0' in df_mask.columns: df_mask.drop(columns=['Unnamed: 0'])

                    min_len = min(len(df_clean), len(df_dirty), len(df_mask))
                    df_clean, df_dirty, df_mask = df_clean.iloc[:min_len].reset_index(drop=True), df_dirty.iloc[
                        :min_len].reset_index(drop=True), df_mask.iloc[:min_len].reset_index(drop=True)
                    df_mask = df_mask.replace(
                        {'True': True, 'False': False, 1: True, 0: False, '1': True, '0': False}).astype(bool)

                    print("Running Sklearn Baselines...")
                    res_mode = SklearnCorrector(strategy='most_frequent').correct(df_dirty, df_mask)
                    p_m, r_m, f1_m = evaluate_correction_metrics(df_clean, df_dirty, res_mode, df_mask)

                    res_knn = SklearnCorrector(strategy='knn').correct(df_dirty, df_mask)
                    p_k, r_k, f1_k = evaluate_correction_metrics(df_clean, df_dirty, res_knn, df_mask)

                    print("Running ZeroEC...")
                    params.update({
                        'clean_data_path': paths.get('cleanDataPath'),
                        'dirty_data_path': paths.get('dataPath'),
                        'detection_path': paths.get('detectionPath'),
                        'embedding_model_path': paths.get('embeddingModelPath'),
                        'output_dir': paths.get('outputDir'),
                        'prompt_dir': paths.get('promptDir')
                    })
                    zeroec = ZeroEC(**params)
                    raw_corrected = zeroec.train_and_predict()

                    df_corrected = df_dirty.mask(df_mask, raw_corrected)
                    p_o, r_o, f1_o = evaluate_correction_metrics(df_clean, df_dirty, df_corrected, df_mask)

                    result = {
                        "cor_prec_mode": f"{p_m:.2%}", "cor_rec_mode": f"{r_m:.2%}", "cor_f1_mode": f"{f1_m:.2%}",
                        "cor_prec_knn": f"{p_k:.2%}", "cor_rec_knn": f"{r_k:.2%}", "cor_f1_knn": f"{f1_k:.2%}",
                        "cor_prec_ours": f"{p_o:.2%}", "cor_rec_ours": f"{r_o:.2%}", "cor_f1_ours": f"{f1_o:.2%}",
                    }

                    result_data = generate_result_data(df_corrected, df_mask)

                else:
                    raise Exception(f"不支持的算法: {method}")

                print("\n✅ 任务全部执行完毕！")
                asyncio.run_coroutine_threadsafe(
                    log_queue.put({"__done__": True, "metrics": result, "result_data": result_data}),
                    loop
                )

            except Exception as e:
                traceback.print_exc()
                asyncio.run_coroutine_threadsafe(log_queue.put({"__error__": str(e)}), loop)

    executor = ThreadPoolExecutor(max_workers=1)
    loop.run_in_executor(executor, run_ml_task)

    try:
        while True:
            msg = await log_queue.get()
            if "__done__" in msg:
                await websocket.send_json({
                    "status": "success",
                    "metrics": msg["metrics"],
                    "result_data": msg.get("result_data")
                })
                break
            elif "__error__" in msg:
                await websocket.send_json({"status": "error", "detail": msg["__error__"]})
                break
            else:
                await websocket.send_json(msg)
    except Exception as e:
        print("WebSocket disconnected:", e)
    finally:
        try:
            await websocket.close()
        except RuntimeError:
            pass
        except Exception:
            pass


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8088, reload=True)