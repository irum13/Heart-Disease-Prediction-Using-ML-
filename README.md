# ❤️ Heart Disease Prediction Using Optimized Feature Engineering and Machine Learning

## 📌 Abstract
Cardiovascular diseases are a growing global concern, leading to an increase in sudden cardiac deaths. Traditional detection methods are slow and often ineffective. This project proposes a **machine learning-based approach** using optimized **feature engineering** to predict the presence and **severity of heart disease**.

A combination of **Principal Component Analysis (PCA)**, **parallel processing**, and **advanced ML algorithms** (including Gradient Boosting and Stacking Classifier) is used to enhance prediction accuracy. The system achieves **100% accuracy** with **0.003s processing time** using the Gradient Boosting Classifier.

---

## ❌ Existing System

Heart disease detection systems currently face limitations like:

- Use of **limited or imbalanced datasets** (e.g., UCI dataset with only 303 entries).
- **Lack of optimized feature selection**, leading to overfitting or underfitting.
- **Poor generalization** to unseen patient data.
- **Limited severity-level classification**, focusing only on binary prediction.
- **Slow processing speed** due to lack of parallelization.

---

## ✅ Proposed System

Our proposed system addresses these issues with:

- A **combined dataset (1025 records)** from Kaggle (Cleveland, Hungarian, Statlog, Long Beach).
- **Data cleaning**, **exploratory analysis**, and **correlation heatmaps**.
- **Feature Engineering Pipeline**:
  - `StandardScaler`
  - `SelectKBest` (ANOVA F-test)
  - `Principal Component Analysis (PCA)` – reduced to 8 components
- Use of **GridSearchCV** for hyperparameter tuning
- **5-fold cross-validation** for robust evaluation
- **Parallel processing (Joblib)** to reduce training time
- A web-based prediction interface built using **Flask**

---

## 🧠 Algorithms Used

| Algorithm               | Description |
|------------------------|-------------|
| **Logistic Regression** | Linear classifier using sigmoid function for binary output |
| **Naive Bayes**         | Probabilistic classifier using Gaussian distribution |
| **Decision Tree**       | Tree-based model that splits based on Gini or Entropy |
| **Random Forest**       | Ensemble of Decision Trees using bagging |
| **Gradient Boosting**   | Sequential ensemble model using boosting |
| **Stacking Classifier** | Combines RF and GB as base models, LR as meta-model |

---

## 🏗️ System Architecture

- 📂 Data Collection & Cleaning
- 📊 Exploratory Data Analysis
- 🧹 Feature Selection & Transformation
- 🤖 Model Training & Cross-Validation
- 🧪 Model Testing on Unseen Data
- 🔥 Severity Score using `predict_proba`
- 🌐 Web-based Interface via Flask

---

## 📊 Results

| Model               | Accuracy | Precision | Recall | F1-Score | Training Time | Testing Time |
|--------------------|----------|-----------|--------|----------|----------------|---------------|
| Logistic Regression| 86%      | 86%       | 86%    | 86%      | 0.047s         | 0.011s        |
| Naive Bayes        | 81%      | 81%       | 81%    | 81%      | 0.015s         | 0.004s        |
| Decision Tree      | 91%      | 91%       | 91%    | 91%      | 0.013s         | 0.003s        |
| Random Forest      | 97%      | 97%       | 97%    | 97%      | 0.189s         | 0.011s        |
| **Gradient Boosting** | **100%** | **100%**   | **100%**| **100%** | **0.255s**     | **0.003s**    |
| Stacking Classifier| 99%      | 99%       | 99%    | 99%      | 2.094s         | 0.009s        |

> Final model: **Gradient Boosting** achieved 100% accuracy with a testing time of only 0.003s.

## 🧠 Technologies Used

- **Language**: Python
- **Framework**: Flask (for frontend + backend integration)
- **Libraries**: scikit-learn, pandas, numpy, seaborn, matplotlib, plotly, joblib
- **Environment**: Anaconda with JupyterLab

## ⚙️ Setup Instructions

1. Clone the repository:
   ```bash
   git clone (https://github.com/irum13/Heart-Disease-Prediction-Using-ML-/tree/main)

2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Run the model:
   ```sh
   python app.py

## 📖 Heart Disease Prediction Notebook

You can view the full Jupyter Notebook on **Nbviewer** here:  
🔗 [View Notebook](https://nbviewer.org/github/irum13/Heart-Disease-Prediction-Using-ML-/blob/main/ReplaceRows.ipynb)

If you want to run this notebook online, open it in **Google Colab**:  
🚀 [Run on Colab](https://colab.research.google.com/github/irum13/Heart-Disease-Prediction-Using-ML-/blob/main/ReplaceRows.ipynb)

## 📸 Application Screenshots

### 🏠 Home Page
![Home Page](https://github.com/irum13/Heart-Disease-Prediction-Using-ML-/blob/c182b037c546fd4ce056afa147afd02fe8b588ff/Project%20Screenshot.png)

