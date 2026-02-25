# Confidence-Aware Early Ransomware Detection Using Behavioral Analysis
---

## 📌 Overview
This project proposes a **confidence-aware framework for early ransomware detection** using **execution-level behavioral analysis** of Windows Sysmon logs from the **SILRAD dataset**.  

Instead of making rigid binary decisions, the system outputs:

- 🚨 ALERT → High-confidence ransomware  
- ⏳ DEFER → Suspicious but uncertain (needs more monitoring)  
- ✅ BENIGN → Normal execution  

This reduces false positives and improves trust in early-stage detection. :contentReference[oaicite:0]{index=0}

---

## 🎯 Problem Statement
Traditional ransomware detection methods:
- Detect after encryption (too late)
- Use binary classification → high false alarms
- Ignore uncertainty in early behavior

Early-stage detection is challenging because only **partial execution data** is available. :contentReference[oaicite:1]{index=1}

---

## 💡 Proposed Solution
We model each **process execution independently** using the **first K = 50 Sysmon events** and:

1. Represent events using **fastText embeddings**
2. Aggregate into execution-level behavioral features
3. Predict ransomware **risk score** using **LightGBM**
4. Compute **SHAP explanation strength**
5. Apply **confidence-aware decision logic**

Decision rule:

| Risk | Explanation | Decision |
|------|------------|----------|
High | High | ALERT |
High | Low | DEFER |
Low | Any | BENIGN |

This explicitly handles uncertainty. :contentReference[oaicite:2]{index=2}

---

## 🧠 Methodology
- Execution-level modeling (ProcessGuid based)
- Early behavioral window (first 50 events)
- Statistical pooling of embeddings:
  - Mean  
  - Standard deviation  
  - Max  
  - Avg absolute difference  
- LightGBM risk estimation
- SHAP-based explainability
- Confidence-aware triage system :contentReference[oaicite:3]{index=3}

---

## 📊 Dataset
**SILRAD – Streaming Intrusion Learning for Ransomware Attack Detection**

- Windows Sysmon logs  
- Benign + multiple ransomware families  
- Execution-level labels  
- fastText event embeddings provided :contentReference[oaicite:4]{index=4}

---

## 📈 Results

### 🔹 Model Performance

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-------|----------|-----------|--------|----|---------|
LightGBM (Proposed) | 0.876 | 0.829 | 0.560 | 0.669 | 0.749 |
XGBoost | 0.875 | 0.826 | 0.556 | 0.665 | 0.814 |
Random Forest | 0.553 | 0.250 | 0.502 | 0.334 | 0.582 |

LightGBM performs best under early-stage constraints. :contentReference[oaicite:5]{index=5}

---

### 🔹 Confidence-Aware Decisions

| Decision | Executions |
|----------|-----------|
BENIGN | 6168 |
ALERT | 669 |
DEFER | 426 |

Only high-confidence cases are alerted, reducing false alarms. :contentReference[oaicite:6]{index=6}

---

## 📊 Key Observations
- Early ransomware signals exist within first 50 events  
- Explanation strength improves trust in alerts  
- DEFER state prevents premature classification  
- Suitable for practical endpoint security deployment :contentReference[oaicite:7]{index=7}

---

## ⚙️ How to Run

```bash
git clone https://github.com/Prachee314/confidence_aware_early_ransomware_detection.git
cd confidence_aware_early_ransomware_detection
pip install -r requirements.txt
python ui/app.py
