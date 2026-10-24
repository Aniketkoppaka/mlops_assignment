import os
import json
import joblib
import pandas as pd
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, classification_report

def train_and_register_model(data_path: str, model_version: str = "v1_baseline"):
    print(f"[*] Training model ({model_version}) using {data_path}...")
    df = pd.read_csv(data_path)
    X = df.drop(columns=["target"])
    y = df["target"]
    
    # Train Random Forest Classifier
    clf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    clf.fit(X, y)
    
    # Evaluate on training distribution
    y_pred = clf.predict(X)
    y_prob = clf.predict_proba(X)[:, 1]
    
    metrics = {
        "model_version": model_version,
        "training_data": os.path.basename(data_path),
        "trained_at": datetime.now().isoformat(),
        "accuracy": float(accuracy_score(y, y_pred)),
        "roc_auc": float(roc_auc_score(y, y_prob)),
        "f1_score": float(f1_score(y, y_pred)),
        "feature_count": X.shape[1],
        "training_samples": len(df)
    }
    
    models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    os.makedirs(models_dir, exist_ok=True)
    
    # Save model weights
    model_file = os.path.join(models_dir, f"model_{model_version}.joblib")
    active_model_file = os.path.join(models_dir, "active_model.joblib")
    metadata_file = os.path.join(models_dir, "model_registry.json")
    
    joblib.dump(clf, model_file)
    joblib.dump(clf, active_model_file) # Set as currently active in production
    
    # Update registry
    registry = []
    if os.path.exists(metadata_file):
        try:
            with open(metadata_file, "r") as f:
                registry = json.load(f)
        except Exception:
            registry = []
            
    registry.append(metrics)
    with open(metadata_file, "w") as f:
        json.dump(registry, f, indent=4)
        
    print(f" Model saved to: {model_file}")
    print(f" Set as active model: {active_model_file}")
    print(f" Metrics: Accuracy={metrics['accuracy']:.4f}, ROC-AUC={metrics['roc_auc']:.4f}, F1={metrics['f1_score']:.4f}")
    return clf, metrics

if __name__ == "__main__":
    base_data = os.path.join(os.path.dirname(__file__), "..", "data", "reference_train.csv")
    train_and_register_model(base_data, "v1_baseline")
