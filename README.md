# Credit Card Default Prediction

A machine learning project comparing Logistic Regression and Support Vector Machine (SVM) models for predicting credit card defaults, with emphasis on handling class imbalance using SMOTE.

## 📋 Overview

This project analyzes credit card default prediction using supervised learning models. The dataset contained significant class imbalance between defaulters and non-defaulters, which was addressed using Synthetic Minority Over-sampling Technique (SMOTE). The study evaluates the impact of oversampling on model performance and compares different SVM kernels.

## 🎯 Objectives

- Compare Logistic Regression and SVM models for credit card default prediction
- Address class imbalance using SMOTE
- Evaluate trade-offs between predictive accuracy and model interpretability
- Analyze impact of different SVM kernels (Linear vs RBF)

## 📊 Dataset

- **Target Variable**: Credit card default (binary classification)
- **Class Imbalance**: Significant imbalance with non-defaulters heavily outnumbering defaulters
- **Preprocessing**: 
  - Feature scaling using StandardScaler
  - SMOTE applied for class balancing

## 🔧 Methodology

### 1. Data Preprocessing
- Checked for missing values
- Identified class imbalance
- Applied SMOTE to generate synthetic minority class samples
- Performed feature scaling using StandardScaler

### 2. Model Training
- **Logistic Regression**: Baseline linear model
- **SVM (Linear Kernel)**: Linear decision boundary
- **SVM (RBF Kernel)**: Non-linear decision boundary
- Hyperparameter optimization using GridSearchCV

### 3. Evaluation Metrics
- Accuracy
- Precision and Recall
- F1-Score
- ROC-AUC Score
- Confusion Matrix
- Decision Boundary Visualization using PCA

## 📈 Results

### Without SMOTE (Imbalanced Data)
| Model | Accuracy | Recall (Defaulters) |
|-------|----------|---------------------|
| Logistic Regression | 80.8% | 24% |
| SVM (RBF Kernel) | 81.4% | 33% |

**Issue**: High accuracy but poor detection of defaulters due to class imbalance

### With SMOTE (Balanced Data)
| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression | 71.98% | ~72% | ~72% | ~72% |
| SVM (Linear Kernel) | 71.96% | ~72% | ~72% | ~72% |
| **SVM (RBF Kernel)** | **77.81%** | **77%** | **76%** | **~78%** |

### Best Model: SVM with RBF Kernel
- **Accuracy**: 77.8%
- **AUC Score**: 0.85 (vs 0.79 for Logistic Regression)
- **Balanced precision-recall** for both classes
- **Improved recall** for defaulters from 24% to 76%

## 🔍 Key Findings

1. **Impact of SMOTE**: Dramatically improved recall for minority class (defaulters) from 24% to 76%

2. **Model Comparison**: 
   - Linear models (Logistic Regression, SVM Linear) performed similarly
   - SVM with RBF kernel captured complex non-linear patterns, achieving superior performance

3. **Trade-offs**:
   - Logistic Regression: More interpretable but lower accuracy
   - SVM (RBF): Higher predictive power but reduced interpretability

4. **Decision Boundaries**: PCA visualization showed SVM RBF kernel created flexible, non-linear decision regions vs linear boundary of Logistic Regression

## 🛠️ Technologies Used

- **Python 3.x**
- **scikit-learn**: Model implementation and evaluation
- **imbalanced-learn**: SMOTE implementation
- **pandas**: Data manipulation
- **numpy**: Numerical computations
- **matplotlib/seaborn**: Visualization
- **GridSearchCV**: Hyperparameter tuning

### Requirements
```
scikit-learn>=1.0.0
imbalanced-learn>=0.9.0
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.4.0
seaborn>=0.11.0
```

## 🚀 Future Improvements

- Gather higher quality data without inherent class imbalance
- Implement XGBoost classifier for enhanced performance and feature importance analysis
- Apply feature selection techniques to refine input variables
- Explore ensemble methods combining multiple models
- Deploy model as web application for real-time predictions

## 📝 Conclusion

This project demonstrates the critical importance of handling class imbalance in financial risk modeling. While logistic regression offers better interpretability, SVM with RBF kernel provides superior predictive accuracy. The application of SMOTE significantly improved the identification of potential defaulters, making the model more effective for real-world risk management applications.

