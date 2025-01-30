# Module 3

## **Overview**
Module 3 of the **AIO24** course introduces fundamental Machine Learning algorithms along with powerful ensemble learning techniques. This module focuses on supervised and unsupervised learning approaches, decision-making models, and boosting techniques to enhance predictive performance.

## **Topics Covered**

### **1. Pandas**
- Introduction to **Pandas**: Data manipulation and analysis library in Python.
- **Key Functions**: DataFrames, Series, indexing, filtering, and handling missing data.
- **Data Cleaning & Preprocessing**: Removing duplicates, handling NaN values, and transforming data.
- **Exploratory Data Analysis (EDA)**: Descriptive statistics, groupby, and visualization with Pandas.

### **2. K-Nearest Neighbors (KNN)**
- **Supervised Learning Algorithm**: Used for classification and regression.
- **Working Principle**: Predicts the label based on the majority vote of its 'K' nearest neighbors.
- **Distance Metrics**: Euclidean, Manhattan, Minkowski distances.

### **3. K-Means Clustering**
- **Unsupervised Learning Algorithm**: Used for grouping similar data points.
- **How It Works**: Assigns points to clusters based on centroids and iteratively updates them.

### **4. Decision Tree**
- **Tree-Based Supervised Learning Algorithm** (for classification and regression).
- **Structure**: Root node, decision nodes, and leaf nodes.
- **Splitting Criteria**: Gini Impurity, Entropy (for classification), Mean Squared Error (for regression).

### **5. Random Forest**
- **Ensemble Learning Method**: Collection of multiple decision trees.
- **Bootstrap Aggregation (Bagging)**: Reduces variance by training trees on random subsets of data.
- **Feature Selection**: Uses randomness to improve generalization.

### **6. AdaBoost**
- **Boosting Technique**: Sequentially improves weak classifiers by adjusting sample weights.
- **How It Works**: Misclassified points get higher weights, forcing the next classifier to focus on them.
- **Common Base Estimator**: Decision Stumps (one-level Decision Trees).

### **7. Gradient Boosting (GBM)**
- **Boosting Approach**: Minimizes errors by fitting new models on residuals of previous models.
- **Loss Functions**: MSE (regression), Log Loss (classification).
- **Gradient Descent Optimization**: Guides model adjustments.


### **8. XGBoost (Extreme Gradient Boosting)**
- **Optimized Version of Gradient Boosting**: Faster and more accurate.
- **Key Features**:
  - Handles **missing values automatically**.
  - **Regularization (L1/L2)**: Prevents overfitting.
  - **Parallel Processing**: More efficient than traditional boosting.
  - **Tree Pruning**: Uses max depth rather than depth-wise growth.
