import json
import pandas as pd

def load_data():
    with open("CUADv1.json", "r", encoding="utf-8") as f:
        cuad_gt = json.load(f)["data"]
        
    with open("extracted_clauses.json", "r", encoding="utf-8") as f:
        our_preds = json.load(f)
        
    return cuad_gt, our_preds

def evaluate():
    print("[SEARCH] Booting Ground-Truth Evaluation Matrix against expert CUAD Annotations...")
    cuad_gt, our_preds = load_data()
    
    # Map ground truth by Document Title
    gt_map = {}
    for doc in cuad_gt:
        title = doc["title"]
        if not doc.get("paragraphs"): continue
        qas = doc["paragraphs"][0]["qas"]
        
        truth_dict = {}
        for qa in qas:
            # CUAD perfectly formats their question strings, we dynamically strip it
            q_text = qa["question"].replace('Highlight the parts (if any) of this contract related to "', '').replace('" that should be extracted.', '').strip()
            
            exists = not qa.get("is_impossible", True)
            if len(qa.get("answers", [])) > 0:
                exists = True
                
            truth_dict[q_text.lower()] = exists
            
        gt_map[title] = truth_dict

    # Track Mathematical Evaluation Metrics
    metrics = {}
    
    for contract in our_preds:
        title = contract["contract_title"]
        gt_clauses = gt_map.get(title, {})
        
        for clause in contract["clauses"]:
            cat_name = clause["category_name"]
            
            # Cast the buggy predictions strings into true booleans securely
            pred_exists = clause["exists"]
            if isinstance(pred_exists, str): pred_exists = pred_exists.lower() == 'true'
            
            # Substring matching to perfectly align our 10 requests with CUAD's 41 expert labels
            gt_exists = False
            for gt_key, gt_val in gt_clauses.items():
                if cat_name.lower() in gt_key or gt_key in cat_name.lower():
                    gt_exists = gt_val
                    break
                    
            if cat_name not in metrics:
                metrics[cat_name] = {"TP": 0, "FP": 0, "FN": 0, "TN": 0}
                
            # Compute Verification Matrices
            if pred_exists and gt_exists:
                metrics[cat_name]["TP"] += 1
            elif pred_exists and not gt_exists:
                metrics[cat_name]["FP"] += 1
            elif not pred_exists and gt_exists:
                metrics[cat_name]["FN"] += 1
            else:
                metrics[cat_name]["TN"] += 1
                
    # Calculate Final Precision and Recall
    results = []
    total_tp = total_fp = total_fn = 0
    
    for clause, m in metrics.items():
        tp, fp, fn, tn = m["TP"], m["FP"], m["FN"], m["TN"]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        total_tp += tp
        total_fp += fp
        total_fn += fn
        
        results.append({
            "Clause Category": clause,
            "Precision": f"{precision*100:.1f}%",
            "Recall": f"{recall*100:.1f}%",
            "F1-Score": f"{f1*100:.1f}%",
            "TP": tp, "FP": fp, "FN": fn
        })
        
    df = pd.DataFrame(results)
    
    # Calculate Macro Global Average
    macro_p = total_tp / (total_tp + total_fp) if (total_tp+total_fp) > 0 else 0
    macro_r = total_tp / (total_tp + total_fn) if (total_tp+total_fn) > 0 else 0
    
    print("\n" + "="*80)
    print("[GOLD] EXPERT GROUND TRUTH EVALUATION METRICS (CUAD v1)")
    print("="*80)
    print(df.to_string(index=False))
    print("-" * 80)
    print(f"Overall Model Precision: {macro_p*100:.2f}% (How accurate our Extractions were)")
    print(f"Overall Model Recall:    {macro_r*100:.2f}% (How brilliantly we avoided missing clauses)")
    print("="*80)

if __name__ == "__main__":
    evaluate()
