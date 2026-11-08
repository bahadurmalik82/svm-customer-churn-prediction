# Customer Churn Prediction using Support Vector Machines

This repository contains an end-to-end machine learning analysis focused on predicting customer churn using Support Vector Machines (SVM). The project explores the impact of feature scaling, compares different SVM kernels, visualizes decision boundaries, and systematically tunes hyperparameters to optimize for business value.

## Dataset
* **Source:** Telco Customer Churn dataset from Kaggle.
* **Target:** `Churn` (Binary: 0 or 1).
* **Preprocessing:** Includes mapping categorical variables, one-hot encoding, handling missing numerical values, and applying `StandardScaler`.

## Key Features & Methodology
* **Feature Scaling Impact:** Demonstrates the severe training time penalty and performance drop when training an SVM on unscaled data versus scaled data.
* **Kernel Comparison:** Evaluates Linear, RBF, and Polynomial kernels based on Accuracy, Precision, Recall, F1-score, and Support Vector count.
* **Margin Visualization:** Uses PCA to project features into 2D space, visualizing how the RBF decision boundary and margin adapt as the regularization parameter `C` changes.
* **Hyperparameter Tuning:** Utilizes `GridSearchCV` with 5-fold cross-validation to find the optimal combination of `C` and `gamma` for the RBF kernel.
* **Handling Class Imbalance:** Addresses the business cost of false negatives (missed churners) by utilizing `class_weight='balanced'`, significantly boosting recall.
* **Algorithm Showdown:** Compares the fine-tuned SVM against Decision Trees, Random Forests, Logistic Regression, and KNN.

## Results
The final recommended model for production is the **RBF-kernel SVM with `class_weight='balanced'` (C=10, gamma=0.01)**.
* **Recall:** 0.775
* **F1-Score:** 0.6125

This configuration was chosen because missing an actual churner (false negative) is more costly to the business than offering a retention incentive to a loyal customer (false positive). The balanced SVM successfully caught the most actual churners while maintaining the highest overall F1-score among all tested models.

## How to run

```bash
pip install pandas numpy scikit-learn matplotlib
jupyter notebook "SVM_Template_(1).ipynb"
```

Get the Telco Customer Churn dataset from Kaggle and place it next to the notebook — the dataset file is not committed to this repo.

## Tech Stack
* **Language:** Python 3.11
* **Libraries:** Pandas, NumPy, Scikit-Learn, Matplotlib

## Author
* **Name:** Bahadur Ali Meer Mohammad
* **Role:** Machine Learning Engineer / Agent Builder (LLM Engineer), BS Computer Science (AI) @ Sukkur IBA University
* **LinkedIn:** https://www.linkedin.com/in/bahadur-ali82/

> Note: an earlier version of this README carried a different author name by mistake. This repository is maintained by Bahadur Ali. If this work was done with teammates, add them here with their correct names.
