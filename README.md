# Machine Learning Fundamentals — Graded Assessment
 
This repository contains the complete submission for the Machine Learning Fundamentals Graded Assessment. It is divided into two distinct sections: practical **Python Coding** applying fundamental machine learning models, and **Business Case Analysis** involving strategic data-driven consulting for retail promotions.

---

## 📂 Repository Structure

```text
ml-assessment/
├── part_a/
│   ├── q1_supervised.ipynb          # Heart Disease Classification Models
│   ├── q2_unsupervised.ipynb        # Customer Segmentation via K-Means & PCA
│   └── q3_feature_engineering.ipynb # Retail Promotions Scikit-Learn Pipeline
├── part_b/
│   └── business_analysis.md         # Strategic answers to Business Scenarios
├── data/                            # Raw datasets (CSV)
│   ├── q1_heart_disease.csv
│   ├── q2_customers.csv
│   └── q3_retail_promotions.csv
└── README.md
```

---

## 🛠️ Part A: Python Coding 
*All Jupyter notebooks have been fully executed with dependencies embedded directly in the cells.*

### [Q1. Supervised Learning — Heart Disease Diagnostics](./part_a/q1_supervised.ipynb)
- **Objective:** Build classification models predicting the presence of Heart Disease (`1 = Disease, 0 = Absent`).
- **Methodology:** Handled missing data by merging statistical median & mode imputations. Encoded categorical parameters and trained a Decision Tree, Random Forest, and Gradient Boosting Classifier. 
- **Findings:** Tuned maximum metrics selecting models based heavily on **F1-Score** and **Recall** due to the high-stakes cost of missing a true medical positive diagnosis.

### [Q2. Unsupervised Learning — Customer Segmentation](./part_a/q2_unsupervised.ipynb)
- **Objective:** Utilize clustering analytics to partition customer purchasing trends.
- **Methodology:** Applied strict mathematical distance scaling via `StandardScaler`, deployed K-Means clustering isolating the optimal $K$ boundary utilizing Elbow plotting tests. Reduced dimensionality computationally for visualization via **PCA (Principal Component Analysis)**.
- **Findings:** Successfully synthesized 2D plots separating customers into distinct, actionable business segments (e.g. Budget Shoppers, High Value Loyalists).

### [Q3. Feature Engineering & Regression — Retail Models](./part_a/q3_feature_engineering.ipynb)
- **Objective:** Predict chronological total `items_sold` across disparate geographical retail storefronts.
- **Methodology:** Overcame data leakage issues by designing heavy **Temporal Train-Test Split** boundaries (Historical 80% Train, Recent 20% Test). Abstracted entire preparation and algorithm loading safely within robust scikit-learn `Pipelines` utilizing `ColumnTransformers`.
- **Findings:** Explored relationships mapping performance matrices (RMSE and MAE) across Linear Regressions and Random Forests. Charted results utilizing custom parity models.

---

## 📊 Part B: Business Case Analysis

### [Retail Promotion Effectiveness Strategy](./part_b/business_analysis.md)
Contains a written technical walkthrough bridging algorithmic data science directly to enterprise strategy:
1. **Problem Formulation:** Why targeting standard volumes (`items_sold`) is vastly superior to predicting un-normalized raw revenue. Transitioning from global models to Hierarchical Mixed-Effects modeling protocols.
2. **EDA & Data Architecture:** Designing the strict inner table joins outlining optimal grains while neutralizing highly skewed non-promotional transaction imbalances. 
3. **MLOps Deployment Flow:** Out-Of-Time Validation principles and utilizing `SHAP` matrices bridging raw numeric calculations to marketing director communication.

---

## 🔧 Technologies Used
* **Languages:** `Python 3`
* **Libraries:** `pandas`, `scikit-learn`, `numpy`
* **Data Visualization:** `matplotlib`, `seaborn`
* **Environments:** `Jupyter Notebook`
