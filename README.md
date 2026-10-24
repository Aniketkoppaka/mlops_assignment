# Automated Concept & Data Drift Detection Pipeline with Continuous Model Retraining

**Course:** AIM4142 MLOps — Assignment  
**Student Name:** Aniket Veer Koppaka  
**Registration Number:** 23FE10CAI00423  
**Section:** AIML-B  
**Department:** Department of AI & ML, School of Computer Science and Engineering, Manipal University Jaipur  

---

## 📌 Project Overview
In production machine learning systems, incoming live data distributions inevitably change over time due to seasonal shifts, policy alterations, macroeconomic variations, or user behavior evolutions (**Concept and Covariate Drift**). 

This repository implements a **closed-loop Level 2 MLOps pipeline** that continuously monitors inference payloads, detects distribution drift using rigorous statistical hypothesis tests (Two-Sample Kolmogorov-Smirnov test, Population Stability Index, and Wasserstein Distance), and automatically triggers model retraining and registration without manual intervention.

---

## 🏛️ System Architecture

```
                 +-----------------------------------------+
                 |   Incoming Live Production Traffic      |
                 +-----------------------------------------+
                                      |
                                      v
                 +-----------------------------------------+
                 |       Statistical Drift Detector        |
                 |  • KS-Test (α=0.05)                     |
                 |  • Population Stability Index (PSI>0.2) |
                 |  • Wasserstein Distance                 |
                 +-----------------------------------------+
                                      |
                     +----------------+----------------+
                     |                                 |
              [Drift < 30%]                     [Drift >= 30%]
                     |                                 |
                     v                                 v
         +-----------------------+         +-----------------------+
         | Status: Normal        |         | [ALERT] Retraining    |
         | Continue Serving      |         | Workflow Triggered    |
         +-----------------------+         +-----------------------+
                                                       |
                                                       v
                                           +-----------------------+
                                           | Automated Model Train |
                                           | & Registry Promotion  |
                                           +-----------------------+
                                                       |
                                                       v
                                           +-----------------------+
                                           |  FastAPI Serving API  |
                                           |  (Active Model v2)    |
                                           +-----------------------+
```

---

## 📂 Repository Structure

```
.
├── data/                         # Training baselines & production batches
│   ├── reference_train.csv       # Baseline training set (5,000 samples)
│   ├── batch_stable.csv          # Live batch without drift
│   ├── batch_drifted.csv         # Live batch with simulated macroeconomic drift
│   └── updated_retrained_dataset.csv # Dataset used for retraining
├── models/                       # Model artifact registry
│   ├── active_model.joblib       # Currently deployed production model
│   ├── model_v1_baseline.joblib  # Version 1 baseline model
│   ├── model_v2_retrained_*.joblib # Version 2 retrained model
│   └── model_registry.json       # Version metadata and evaluation metrics
├── reports/                      # Visual artifacts & audit summaries
│   ├── drift_distribution_plot.png
│   ├── latest_drift_report.html  # Interactive visual dashboard
│   └── drift_summary_*.json      # Raw statistical test metrics
├── src/
│   ├── dataset_generator.py      # Generates reference data & drift batches
│   ├── drift_detector.py         # Statistical KS-test & PSI testing engine
│   ├── model_trainer.py          # Random Forest training & registry logger
│   ├── pipeline_runner.py        # Automated trigger & continuous retraining
│   └── app.py                    # Real-time FastAPI inference microservice
├── Project_Report.pdf            # Comprehensive 2-page project report
├── requirements.txt              # Project dependencies
└── README.md
```

---

## 🚀 How to Run the Project

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate Datasets
Creates the baseline dataset and live production batches (with and without drift):
```bash
python src/dataset_generator.py
```

### 3. Train Initial Baseline Model (v1)
Trains the initial model and registers it to `models/active_model.joblib`:
```bash
python src/model_trainer.py
```

### 4. Run Continuous Monitoring & Automated Retraining Pipeline
Simulate live traffic and evaluate drift:
```bash
# Test with drifted traffic (automatically detects drift & triggers retraining to v2):
python src/pipeline_runner.py batch_drifted.csv

# Test with normal stable traffic:
python src/pipeline_runner.py batch_stable.csv
```

### 5. Launch FastAPI Model Serving Microservice
Expose the real-time inference API:
```bash
uvicorn src.app:app --host 127.0.0.1 --port 8000 --reload
```
Open your browser to test endpoints interactively via Swagger UI:
- **Swagger Documentation:** `http://127.0.0.1:8000/docs`
- **Health Check:** `http://127.0.0.1:8000/health`
- **Interactive Drift Report:** Open `reports/latest_drift_report.html` directly in any web browser.

---

## 📊 Experimental Results & Evaluation

| Pipeline Metric | Baseline Model (`v1`) | Retrained Model (`v2`) |
| :--- | :--- | :--- |
| **Training Sample Size** | 5,000 observations | 4,500 observations (sliding window) |
| **Accuracy Score** | 93.02% | 88.04% (adapted to new domain) |
| **ROC-AUC Score** | 0.9758 | 0.9627 |
| **F1-Score** | 0.7955 | 0.8548 |
| **Drift Ratio Detected** | 0.0% (Stable) | 100.0% (Automated Retraining Triggered) |
| **Decision Gate Action**| Continue Serving | Auto-Retrain & Promote Model |

---

## 🔬 Key MLOps Concepts Demonstrated
- **Covariate Shift Detection:** Unsupervised detection on $P(X)$ before ground-truth labels $y$ are delayed.
- **Population Stability Index (PSI):** Industry standard test ($PSI > 0.20$ signals severe drift).
- **Two-Sample Kolmogorov-Smirnov Test:** Non-parametric hypothesis testing ($\alpha = 0.05$).
- **Level 2 MLOps Automation:** Closed-loop event-driven retraining pipeline without human bottlenecks.
- **Model Registry & Governance:** Automated artifact serialization, metric auditing, and version tracking.
