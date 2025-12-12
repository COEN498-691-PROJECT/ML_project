# Investigation of Five Machine Learning Models for Human Activity Recognition
 ## Table of Contents

- [Getting Started](#getting-started)
  - [Installation](#installation)
  - [Collaborators](#collaborators)

## Getting Started

To get this project up and running on your local machine, please follow these steps.

### Installation

1.  **Clone the repository**
    ```bash
    git clone https://github.com/COEN498-691-PROJECT/ML_project.git

2.  **Navigate to the project directory**
    ```bash
    cd ML_project
    ```

3.  **Install dependencies**
    ```bash
    pip install
    ```
### Collaborators

| Name   | Username |
| -------- | ------- |
| Laurie Anne Laberge  | P4SS3-P4RT0UT   |
| Laura Hang | ssugarcane    |
| Yunxing Tao   | aurora-tao   |
| Mahsa Khatibi   | thisisminecraftvillager   |


---

## 🔵 Logistic Regression — Training, Tuning & Conclusion

For the Logistic Regression model, I first prepared the dataset using a participant-based split (GroupShuffleSplit) to ensure no subject overlap between training and testing sets. A preprocessing pipeline with StandardScaler and LogisticRegression was trained using GroupKFold to avoid data leakage and encourage cross-subject generalization.

Hyperparameter tuning was performed with GridSearchCV over different values of the regularization parameter C. The best configuration was **C = 10**, which produced the highest macro F1-score. After tuning, the final model was retrained and evaluated using both a held-out participant split and LOSO (Leave-One-Subject-Out) evaluation.

**Conclusion:** Logistic Regression delivered strong performance (≈0.99 accuracy), stable LOSO results, and high interpretability via feature coefficients, making it a reliable lightweight baseline for HAR tasks.


## 🔴 Support Vector Machine (SVM) — Training, Tuning & Conclusion

The SVM model used the same preprocessing pipeline with StandardScaler, trained with GroupKFold cross-validation to ensure subject-level separation. Hyperparameter tuning via GridSearchCV explored different values of C and gamma for the RBF kernel. The best-performing parameters were **C = 10** and **gamma = 0.01**.

The final SVM model was retrained using these optimal parameters and evaluated on both the subject-independent test split and LOSO. SVM achieved strong generalization with macro F1-scores around 0.93–0.95 during CV, showing solid performance on unseen participants.

**Conclusion:** SVM captured more complex nonlinear patterns than LR, offering competitive performance across subjects at the cost of higher computation, making it a powerful model when accuracy is prioritized over interpretability.



