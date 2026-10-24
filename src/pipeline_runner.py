import os
import json
import pandas as pd
from datetime import datetime
from drift_detector import detect_drift_custom, generate_interactive_html_report
from model_trainer import train_and_register_model

def run_mlops_monitoring_pipeline(current_batch_filename: str = "batch_drifted.csv"):
    print("\n" + "="*70)
    print("      CONTINUOUS MODEL MONITORING & AUTOMATED RETRAINING PIPELINE      ")
    print("="*70)
    
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_dir = os.path.join(project_root, "data")
    reports_dir = os.path.join(project_root, "reports")
    os.makedirs(reports_dir, exist_ok=True)
    
    ref_path = os.path.join(data_dir, "reference_train.csv")
    cur_path = os.path.join(data_dir, current_batch_filename)
    
    print(f"[*] Reference Baseline: {ref_path}")
    print(f"[*] Incoming Inference Batch: {cur_path}")
    
    ref_df = pd.read_csv(ref_path)
    cur_df = pd.read_csv(cur_path)
    
    feature_cols = [c for c in ref_df.columns if c != "target"]
    
    # 1. Run statistical drift tests
    print("\n[Step 1] Executing Two-Sample Kolmogorov-Smirnov & PSI Drift Tests...")
    drift_results = detect_drift_custom(
        ref_df=ref_df,
        cur_df=cur_df,
        feature_cols=feature_cols,
        p_val_threshold=0.05,
        psi_threshold=0.20
    )
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_html = os.path.join(reports_dir, f"drift_report_{timestamp}.html")
    report_latest = os.path.join(reports_dir, "latest_drift_report.html")
    summary_json = os.path.join(reports_dir, f"drift_summary_{timestamp}.json")
    
    # 2. Generate interactive visual dashboards
    print("\n[Step 2] Generating Interactive Visual Reports...")
    generate_interactive_html_report(ref_df, cur_df, drift_results, report_html)
    generate_interactive_html_report(ref_df, cur_df, drift_results, report_latest)
    
    with open(summary_json, "w") as f:
        json.dump(drift_results, f, indent=4)
        
    print(f" Summary JSON saved: {summary_json}")
    
    # 3. Decision Gate: Automated Continuous Retraining
    print("\n[Step 3] Automated Decision Gate (Level 2 MLOps Continuous Training):")
    drifted_count = len(drift_results["drifted_features"])
    total_count = len(feature_cols)
    print(f" -> Drift Ratio: {drift_results['drift_ratio']*100:.1f}% ({drifted_count}/{total_count} features drifted)")
    
    if drift_results["drift_detected"]:
        print("\n [ALERT] SIGNIFICANT DATA DRIFT DETECTED!")
        print(f" Drifted Features: {', '.join(drift_results['drifted_features'])}")
        print(" [TRIGGER] Initiating Automated Retraining Workflow...")
        
        # Merge latest batch with training baseline to form updated training set
        updated_train_df = pd.concat([ref_df.tail(3000), cur_df], ignore_index=True)
        updated_data_path = os.path.join(data_dir, "updated_retrained_dataset.csv")
        updated_train_df.to_csv(updated_data_path, index=False)
        
        new_version = f"v2_retrained_{timestamp}"
        new_model, metrics = train_and_register_model(updated_data_path, model_version=new_version)
        print(f"\n Retraining completed successfully!")
        print(f" -> Active Production Model Updated to: {new_version}")
        print(f" -> New Accuracy: {metrics['accuracy']:.4f}, ROC-AUC: {metrics['roc_auc']:.4f}")
    else:
        print("\n [STATUS] System Stable. Feature distributions are within acceptable tolerances.")
        print(" No retraining necessary. Continuing production inference.")
        
    print("\n" + "="*70)
    print("PIPELINE EXECUTION COMPLETE")
    print("="*70)
    return drift_results

if __name__ == "__main__":
    import sys
    batch = sys.argv[1] if len(sys.argv) > 1 else "batch_drifted.csv"
    run_mlops_monitoring_pipeline(batch)
