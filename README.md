# Customer Churn Prediction using Support Vector Machines

This repository contains an end-to-end machine learning analysis focused on predicting customer churn using Support Vector Machines (SVM). The project explores the impact of feature scaling, compares different SVM kernels, visualizes decision boundaries, and systematically tunes hyperparameters to optimize for business value.[cite: 1]

## Dataset
* **Source:** Telco Customer Churn dataset from Kaggle.[cite: 1]
* **Target:** `Churn` (Binary: 0 or 1).[cite: 1]
* **Preprocessing:** Includes mapping categorical variables, one-hot encoding, handling missing numerical values, and applying `StandardScaler`.[cite: 1]

## Key Features & Methodology
* **Feature Scaling Impact:** Demonstrates the severe training time penalty and performance drop when training an SVM on unscaled data versus scaled data.[cite: 1]
* **Kernel Comparison:** Evaluates Linear, RBF, and Polynomial kernels based on Accuracy, Precision, Recall, F1-score, and Support Vector count.[cite: 1]
* **Margin Visualization:** Uses PCA to project features into 2D space, visualizing how the RBF decision boundary and margin adapt as the regularization parameter `C` changes.[cite: 1]
* **Hyperparameter Tuning:** Utilizes `GridSearchCV` with 5-fold cross-validation to find the optimal combination of `C` and `gamma` for the RBF kernel.[cite: 1]
* **Handling Class Imbalance:** Addresses the business cost of false negatives (missed churners) by utilizing `class_weight='balanced'`, significantly boosting recall.[cite: 1]
* **Algorithm Showdown:** Compares the fine-tuned SVM against Decision Trees, Random Forests, Logistic Regression, and KNN.[cite: 1]

## Results
The final recommended model for production is the **RBF-kernel SVM with `class_weight='balanced'` (C=10, gamma=0.01)**.[cite: 1] 
* **Recall:** 0.775[cite: 1]
* **F1-Score:** 0.6125[cite: 1]

This configuration was chosen because missing an actual churner (false negative) is more costly to the business than offering a retention incentive to a loyal customer (false positive).[cite: 1] The balanced SVM successfully caught the most actual churners while maintaining the highest overall F1-score among all tested models.[cite: 1]

## Tech Stack
* **Language:** Python 3.11[cite: 1]
* **Libraries:** Pandas, NumPy, Scikit-Learn, Matplotlib[cite: 1]

## Author
* **Name:** Marvi Malik[cite: 1]
* **Student ID:** 023-24-0016[cite: 1]
