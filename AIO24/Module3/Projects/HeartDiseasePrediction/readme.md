# **Heart Disease Prediction using Machine Learning**

## **Overview**
Heart disease is a major global health concern, leading to millions of deaths annually. Early detection and accurate prediction of heart disease risk can significantly improve patient outcomes and assist healthcare professionals in decision-making. 

This project leverages **machine learning techniques** to build a predictive model using the **Cleveland Heart Disease dataset** from the **UCI Machine Learning Repository**. The dataset contains various clinical attributes that can help identify individuals at risk of heart disease.

---

## **1. Dataset: Cleveland Heart Disease (UCI)**
The dataset consists of **303 patient records** with **13 clinical attributes**, including:
- **Demographic Information**: Age, Sex
- **Medical History & Symptoms**: Chest Pain Type, Resting Blood Pressure, Cholesterol Levels, Fasting Blood Sugar
- **ECG & Exercise Information**: Resting ECG, Maximum Heart Rate, Exercise-Induced Angina, ST Depression, and Slope of ST Segment
- **Diagnosis**: The target variable indicates the presence or absence of heart disease.

---

## **2. Data Preprocessing**
To ensure optimal model performance, the following preprocessing steps were applied:
- **Handling Missing Values**: Removing or imputing missing data.
- **Data Visualization**: Visualize the data to see the relationship between each category and the target
- **Data Splitting**: Dividing the dataset into **training** and **testing** sets.

---

## **3. Machine Learning Models**
The project explores multiple supervised learning algorithms to identify the most effective model for heart disease prediction:

### **Baseline Model**
- **Naïve Bayes**: A simple probabilistic classifier based on Bayes' theorem.

### **Traditional Models**
- **K-Nearest Neighbors (KNN)**: A distance-based classification algorithm.
- **Decision Tree**: A tree-based model that splits data using decision rules.
- **Support Vector Machine (SVM)**: A margin-based classifier that finds the best decision boundary.

### **Ensemble Learning Methods**
- **Random Forest**: A collection of decision trees that reduces variance.
- **AdaBoost**: An adaptive boosting algorithm that improves weak learners.
- **Gradient Boosting (GBM)**: An optimization-based boosting method.
- **XGBoost**: An efficient and scalable version of gradient boosting.

### **Stacking Ensemble**
A **stacking classifier** is implemented using all the models mentioned above, with **XGBoost** as the **final estimator**. Stacking helps improve prediction accuracy by leveraging the strengths of multiple models.

---

## **4. Model Evaluation**
Each model is evaluated using multiple metrics:
- **Accuracy**: Measures the overall correctness of predictions in terms of train data and test data.
- **Precision & Recall**: Determines the reliability and completeness of positive predictions.
- **F1-Score**: Balances precision and recall.
---

## **6. Future Improvements**
To further enhance the model, the following improvements can be explored:
- **Hyperparameter Tuning**: Optimizing model parameters for better performance.
- **Deep Learning Approaches**: Using neural networks for complex feature interactions.
- **Larger Datasets**: Incorporating more patient records for better generalization.
- **Explainability Methods**: Using SHAP or LIME to interpret model predictions.

---

## **7. Conclusion**
This project successfully applies **machine learning techniques** to predict heart disease risk using the **Cleveland Heart Disease dataset**. The use of **ensemble methods and stacking** demonstrates how combining multiple models enhances predictive accuracy, providing a powerful tool for early diagnosis and risk assessment in the medical field.

---
