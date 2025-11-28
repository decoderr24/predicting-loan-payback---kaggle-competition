# 🏆 Predicting Loan Payback - Kaggle Competition (Top 50)

This repository contains my solution for the **Kaggle Playground Series - Season 5, Episode 11: Predicting Loan Payback**. This solution achieved a competitive score, securing a position in the **Top 50** on the leaderboard (Score: **0.92742**).

🔗 **Competition Link:** [Predicting Loan Payback | Kaggle](https://www.kaggle.com/competitions/playground-series-s5e11)

-----

## 📌 Project Overview

The goal of this competition was to predict the probability of a borrower paying back their loan (`loan_paid_back` target). The dataset consists of synthetically generated data based on real-world financial attributes.

**Key Challenges:**

  * **Imbalanced Dataset:** The target class distribution was skewed.
  * **Noise & Synthetic Artifacts:** Handling synthetic data inconsistencies.
  * **High Competition:** The difference between Top 100 and Top 1000 was extremely marginal (\< 0.0001 AUC).

-----

## 🛠️ Methodology & Strategy

My approach focused on a robust **Ensemble Strategy**, combining diverse Gradient Boosting models with advanced Feature Engineering and Strategic Blending.

[Image of ensemble learning stacking diagram]

### 1\. Feature Engineering ⚙️

To boost model performance, I engineered several domain-specific financial features:

  * **Log Transformations:** Applied to skewed features like `annual_income` and `loan_amount` to normalize distribution.
  * **Financial Ratios:**
      * `Income-to-Loan Ratio`: Measuring borrower's capacity to repay.
      * `Total Interest Payable`: Calculating the total cost of borrowing.
      * `Disposable Income`: Estimating remaining funds after loan obligations.
  * **Interaction Features:** Combining `grade`, `interest_rate`, and `income` to capture non-linear relationships.

### 2\. Modeling Pipeline 🤖

I utilized a diverse set of State-of-the-Art (SOTA) Gradient Boosting Decision Trees (GBDT):

| Model | Library | Role in Ensemble |
| :--- | :--- | :--- |
| **XGBoost** | `xgboost` | High accuracy, handling non-linear interactions effectively. |
| **LightGBM** | `lightgbm` | Fast training speed and excellent handling of large datasets. |
| **CatBoost** | `catboost` | Native handling of categorical features without extensive preprocessing. |

### 3\. Validation Strategy 📊

  * **Stratified K-Fold Cross-Validation (10 Folds):** To ensure model stability and prevent overfitting on the imbalanced dataset.
  * **Metric:** Area Under the ROC Curve (AUC-ROC).

### 4\. Ensemble & Blending ⚗️

The final submission was achieved through a multi-stage blending process:

1.  **Internal Ensemble:** Weighted average of my own XGBoost, LightGBM, and CatBoost models.
2.  **External Blending:** Strategic blending with high-scoring public notebook outputs (using "Power Averaging" and "Rank Averaging" techniques) to refine the decision boundary and push the score into the Top 50.

> **Final Score:** `0.92778` (Public Leaderboard)

<img width="1489" height="1089" alt="output" src="https://github.com/user-attachments/assets/04873d22-753d-4c16-ab7c-adf1834c3c4b" />

-----

## 📂 Repository Structure

```bash
├── predicting-loan-payback.ipynb  # Main notebook containing EDA, FE, and Modeling
├── README.md                      # Project documentation
└── submission_final_optimal.csv   # Final submission file
```

## 🚀 How to Run

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/decoderr24/predicting-loan-payback---kaggle-competition.git
    cd predicting-loan-payback---kaggle-competition
    ```

2.  **Install dependencies:**
    Make sure you have the following libraries installed:

    ```bash
    pip install pandas numpy matplotlib seaborn xgboost lightgbm catboost scikit-learn
    ```

3.  **Run the Notebook:**
    Open `predicting-loan-payback.ipynb` in Jupyter Notebook, Google Colab, or Kaggle Kernels and execute the cells sequentially.

-----

## 📈 Results

  * **Public Leaderboard Score:** \~0.92778
  * **Rank:** Top 50 🏅

-----

## 🤝 Acknowledgements

Special thanks to the Kaggle community for sharing insights and to the hosts for providing an interesting dataset. The blending strategy was inspired by various public kernels and refined with custom feature engineering.


