<!-- @format -->

# Telecom Customer Churn Prediction

Machine learning solution for predicting customer churn in telecommunications using Random Forest classification. Achieves 79% accuracy with 82% ROC-AUC score.

## 💡 Why This Matters

Customer acquisition costs 5-25x more than retention. This model identifies at-risk customers, enabling proactive retention strategies that can boost profits by 25-95% with just 5% better retention.

## 🚀 Quick Start

1. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

2. **Run the notebook:**

   ```bash
   jupyter notebook telecom_churn_prediction.ipynb
   ```

   The dataset loads automatically from GitHub - no manual download needed!

**📊 Dataset Source:** [Telco Customer Churn Dataset (Kaggle)](https://www.kaggle.com/datasets/yeanzc/telco-customer-churn-ibm-dataset)

## Tech Stack

- Python 3.7+ | Pandas, NumPy
- Scikit-learn, imbalanced-learn (SMOTE)
- Matplotlib, Seaborn

## 📈 Model Results

| Metric    | Score |
| --------- | ----- |
| Accuracy  | 0.79  |
| ROC-AUC   | 0.82  |
| Precision | 0.60  |
| Recall    | 0.59  |

## 📂 Project Structure

```
├── telecom_churn_prediction.ipynb    # Main analysis with full pipeline
├── data/Telco_customer_churn.csv     # Dataset (7,043 customers, 33 features)
├── requirements.txt                  # Dependencies
└── README.md
```

## 🔮 What's Inside

- **Data Pipeline**: Cleaning, feature engineering, one-hot encoding
- **Class Imbalance**: SMOTE technique for balanced training
- **Model**: Random Forest (100 trees) with proper validation
- **Analysis**: Feature importance, confusion matrix, business metrics
