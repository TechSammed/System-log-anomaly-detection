# System Log Anomaly Detection (Unsupervised ML)

This project detects abnormal system behavior from large-scale HDFS system logs
using unsupervised machine learning.  
The system learns normal operational patterns and flags statistically rare
log windows as anomalies without using labeled data.

---

## 🔍 Problem Statement

Modern distributed systems generate millions of log entries daily.
Manual monitoring is impractical, and labeled failure data is often unavailable.
This project addresses that challenge by automatically identifying
unusual log patterns that may indicate system instability or failures.

---

## 🧠 Approach

- Parsed raw HDFS log files using regex-based log parsing
- Aggregated logs into fixed-size time windows
- Engineered behavioral features such as:
  - Warning frequency
  - Activity volume
  - Message pattern changes
  - Subsystem diversity
- Applied **Isolation Forest** for unsupervised anomaly detection
- Visualized anomalies using PCA and time-based anomaly scores
- Built an interactive **Streamlit dashboard** for monitoring

---

## 📊 Features Used

| Feature | Description |
|------|------------|
| is_warn | Number of WARN logs per window |
| is_info | Number of INFO logs per window |
| msg_length | Average log message length |
| unique_components | Number of distinct system components |

---

## 🚨 Anomaly Detection

- The model assigns an **anomaly score** to each log window
- More negative scores indicate stronger deviations from normal behavior
- Anomalies are ranked by severity instead of binary failure prediction

---

## 🖥️ Streamlit Dashboard

The deployed dashboard provides:
- Total windows vs anomalous windows
- Anomaly score over time visualization
- PCA-based normal vs anomalous behavior visualization
- Table of most anomalous log windows

---

## 🛠️ Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn (Isolation Forest, PCA)
- Matplotlib
- Streamlit

---

## 🚀 How to Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```
##📁 Project Structure
```
log-anomaly-detection/
├── app.py
├── features.csv
├── isolation_forest.pkl
├── notebooks/
├── requirements.txt
└── README.md
```

