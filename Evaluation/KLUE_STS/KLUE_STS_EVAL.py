import os
import json
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import precision_recall_fscore_support, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split 

def calculate_metrics(true_scores, pred_scores):
    # None 값 제거 및 유효한 점수 쌍 필터링
    valid_pairs = [(t, p) for t, p in zip(true_scores, pred_scores) 
                  if t is not None and p is not None]
    
    if not valid_pairs:
        return {
            "F1_score": None,
            "Pearson_r": None,
            "RMSE": None,
            "MAE": None,
            "MSE": None
        }
    
    true_clean, pred_clean = zip(*valid_pairs)
    
    # F1 Score (임계값 3.0 기준)
    true_binary = [1 if score >= 3.0 else 0 for score in true_clean]
    pred_binary = [1 if score >= 3.0 else 0 for score in pred_clean]
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_binary, pred_binary, average='binary', zero_division=0
    )
    
    # Pearson's r
    try:
        pearson_r, _ = pearsonr(true_clean, pred_clean)
    except:
        pearson_r = None
    
    # RMSE, MAE, MSE
    rmse = np.sqrt(mean_squared_error(true_clean, pred_clean))
    mae = mean_absolute_error(true_clean, pred_clean)
    mse = mean_squared_error(true_clean, pred_clean)
    
    # 4자리 수로 반올림
    return {
        "F1_score": round(float(f1), 4) if f1 is not None else None,
        "Pearson_r": round(float(pearson_r), 4) if pearson_r is not None else None,
        "RMSE": round(float(rmse), 4),
        "MAE": round(float(mae), 4),
        "MSE": round(float(mse), 4)
    }

def process_log_files(root_dir):
    results = {}
    
    for model_dir in os.listdir(root_dir):
        log_path = os.path.join(root_dir, model_dir, "log.json")
        if os.path.exists(log_path):
            try:
                with open(log_path, 'r', encoding='utf-8') as f:
                    logs = json.load(f)
                
                true_scores = []
                pred_scores = []
                for log in logs:
                    true_scores.append(log.get("true_score"))
                    pred_scores.append(log.get("predicted_score"))
                
                metrics = calculate_metrics(true_scores, pred_scores)
                results[model_dir] = metrics
                
            except Exception as e:
                print(f"Error processing {model_dir}: {str(e)}")
                results[model_dir] = {
                    "F1_score": None,
                    "Pearson_r": None,
                    "RMSE": None,
                    "MAE": None,
                    "MSE": None
                }
    
    output_path = os.path.join(root_dir, "STS_results.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    
    print(f"✅ Results saved to {output_path}")
    return results

if __name__ == "__main__":
    root_directory = "/scratch/jsong132/Can_LLM_Learn_New_Language/Evaluation/KLUE_STS/klue_sts_results"
    process_log_files(root_directory)