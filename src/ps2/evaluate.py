import os
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def run_evaluation():
    print("==================================================")
    print("Evaluating Model Performances & Logging Results")
    print("==================================================")

    # Paths
    processed_csv = os.path.join("data", "processed_astram_events.csv")
    nn_oof_path = os.path.join("models", "nn_oof_predictions.pkl")
    gbdt_oof_path = os.path.join("models", "gbdt_oof_predictions.pkl")
    log_path = os.path.join("logs", "run_summary.log")

    if not all(os.path.exists(p) for p in [processed_csv, nn_oof_path, gbdt_oof_path]):
        print("Error: Required files for evaluation are missing. Make sure train.py has run successfully.")
        return

    # Load targets
    df = pd.read_csv(processed_csv)
    targets = {
        'eis': df['target_eis'].values,
        'manpower': df['target_manpower'].values,
        'barricades': df['target_barricades'].values,
        'diversion': df['target_diversion'].values
    }

    # Load Out-Of-Fold Predictions
    with open(nn_oof_path, "rb") as f:
        nn_preds = pickle.load(f)
    with open(gbdt_oof_path, "rb") as f:
        gbdt_preds = pickle.load(f)

    # Compute Ensemble predictions (50-50 average)
    ensemble_preds = {}
    for key in targets.keys():
        ensemble_preds[key] = 0.5 * nn_preds[key] + 0.5 * gbdt_preds[key]

    # Evaluate each target
    results = {}
    for key, target_val in targets.items():
        results[key] = {}
        
        # Regression Targets
        if key in ['eis', 'manpower', 'barricades']:
            for model_name, preds_dict in [('Two-Tower Neural Network', nn_preds), 
                                           ('LightGBM Baseline', gbdt_preds), 
                                           ('Weighted Ensemble (DL+GBDT)', ensemble_preds)]:
                preds = preds_dict[key]
                # Round manpower and barricades for integer counts
                if key in ['manpower', 'barricades']:
                    preds = np.round(preds)

                mae = mean_absolute_error(target_val, preds)
                rmse = np.sqrt(mean_squared_error(target_val, preds))
                r2 = r2_score(target_val, preds)

                results[key][model_name] = {
                    'MAE': mae,
                    'RMSE': rmse,
                    'R2': r2
                }
        
        # Binary Classification Target
        else: # diversion
            for model_name, preds_dict in [('Two-Tower Neural Network', nn_preds), 
                                           ('LightGBM Baseline', gbdt_preds), 
                                           ('Weighted Ensemble (DL+GBDT)', ensemble_preds)]:
                probs = preds_dict[key]
                preds = (probs >= 0.5).astype(int)

                acc = accuracy_score(target_val, preds)
                f1 = f1_score(target_val, preds, zero_division=0)
                prec = precision_score(target_val, preds, zero_division=0)
                rec = recall_score(target_val, preds, zero_division=0)

                results[key][model_name] = {
                    'Accuracy': acc,
                    'F1-Score': f1,
                    'Precision': prec,
                    'Recall': rec
                }

    # Print & Log Summary
    log_content = []
    log_content.append("==================================================")
    log_content.append("ASTraM Traffic Event Intelligence Evaluation Log")
    log_content.append("==================================================")
    
    for key in targets.keys():
        log_content.append(f"\nTarget Variable: {key.upper()}")
        log_content.append("-" * 50)
        
        if key in ['eis', 'manpower', 'barricades']:
            # Header
            log_content.append(f"{'Model Name':<30} | {'MAE':<10} | {'RMSE':<10} | {'R2-Score':<10}")
            log_content.append("-" * 70)
            for model_name, metrics in results[key].items():
                log_content.append(f"{model_name:<30} | {metrics['MAE']:<10.4f} | {metrics['RMSE']:<10.4f} | {metrics['R2']:<10.4f}")
        else:
            # Header
            log_content.append(f"{'Model Name':<30} | {'Accuracy':<10} | {'F1-Score':<10} | {'Precision':<10} | {'Recall':<10}")
            log_content.append("-" * 80)
            for model_name, metrics in results[key].items():
                log_content.append(f"{model_name:<30} | {metrics['Accuracy']:<10.4f} | {metrics['F1-Score']:<10.4f} | {metrics['Precision']:<10.4f} | {metrics['Recall']:<10.4f}")

    # Output to stdout and save to file
    final_output = "\n".join(log_content)
    print(final_output)

    with open(log_path, "w") as f:
        f.write(final_output)
    print(f"\nFinal evaluation summary written to {log_path}")
    print("==================================================")

if __name__ == "__main__":
    run_evaluation()
