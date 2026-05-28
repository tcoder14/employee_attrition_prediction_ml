# Predicting Employee Attrition Using Machine Learning

## Introduction

This project focuses on predicting employee attrition using machine learning models. Employee attrition means employees leaving an organization. The main goal of this project is to identify which employees are more likely to leave so that organizations can take early actions to retain valuable employees.

Human resource departments can use this type of prediction system to better understand the factors behind employee attrition and make data-driven decisions.

## Dataset Description

The dataset contains employee-related information with both numerical and categorical features.

- Total data points: 1,677 rows
- Total features: 35 columns
- Problem type: Classification
- Target variable: `Attrition`
  - `0` = Employee stays
  - `1` = Employee leaves

The dataset contains two main types of features:

- Numerical features, such as:
  - Age
  - Monthly Income
  - Years at Company
  - Total Working Years

- Categorical features, such as:
  - Business Travel
  - Department
  - Job Role

Since machine learning models cannot directly process text values, categorical features were converted into numerical values using Label Encoding.

## Correlation Analysis

Pearson correlation was used to understand the linear relationship between the features and the target variable `Attrition`.

The correlation values range from `-1` to `+1`.

- A positive value means the feature may increase the chance of attrition.
- A negative value means the feature may reduce the chance of attrition.
- A value close to `0` means there is little or no linear relationship.

From the correlation analysis, it was observed that no feature had a very strong correlation with attrition. However, some useful patterns were found:

- `DistanceFromHome` showed a positive relationship with attrition.
  This means employees who live farther from work may be more likely to leave.

- `TotalWorkingYears`, `JobLevel`, and `MonthlyIncome` showed negative relationships with attrition.
  This means senior and well-paid employees may be less likely to leave.

## Class Imbalance

The dataset is highly imbalanced.

- Class `0` / No Attrition: 1,477 employees
- Class `1` / Attrition: 200 employees

This means most employees in the dataset did not leave the organization.

Because of this imbalance, accuracy alone is not a reliable evaluation metric. A model can get high accuracy by mostly predicting the majority class, but it may fail to correctly identify employees who actually leave.

## Data Preprocessing

Several preprocessing steps were applied before training the models.

### 1. Removing Redundant Columns

Some columns had the same value for every row, such as:

- `EmployeeCount`
- `StandardHours`
- `Over18`

These columns were removed because they do not help the model make predictions.

The `id` column was also removed because it does not carry useful predictive information.

### 2. Encoding Categorical Features

Categorical columns contain text values. Since machine learning models work with numbers, these text values were converted into numerical values using Label Encoding.

For example, categories like `Travel_Rarely` and `Travel_Frequently` were converted into integer values.

### 3. Feature Scaling

The numerical features had different value ranges. For example:

- `MonthlyIncome` can have very large values.
- `YearsSinceLastPromotion` usually has smaller values.

To handle this, StandardScaler was used. It transforms the data so that the features have a mean of `0` and a standard deviation of `1`.

This helps models like KNN and Neural Networks perform better because large-value features will not dominate the model.

## Dataset Splitting

The dataset was split into:

- 80% training data
- 20% testing data

A stratified split was used because the dataset is imbalanced. Stratification keeps the same class ratio in both training and testing data.

This ensures that both sets contain a similar percentage of attrition and non-attrition cases.

## Models Used

Three supervised machine learning models were trained and tested.

### 1. K-Nearest Neighbors

K-Nearest Neighbors was used as a baseline model. It predicts the class of a data point based on the classes of nearby data points.

It is simple and useful for checking whether similar employees show similar attrition behavior.

### 2. Decision Tree

Decision Tree was used because it is easy to interpret. It makes decisions by splitting the dataset based on important features.

This model helps identify which features may be important for predicting employee attrition.

### 3. Neural Network

A Multi-Layer Perceptron classifier was used to capture more complex and non-linear relationships in the dataset.

This model can learn deeper patterns from the data compared to simpler models.

## Unsupervised Learning

K-Means clustering was also applied as an unsupervised learning approach.

The number of clusters was set to `2` because the target problem has two possible outcomes:

- Employee stays
- Employee leaves

The goal was to check whether the algorithm could naturally group employees into two clusters without using the actual attrition labels.

## Model Evaluation

The models were evaluated using different metrics:

- Accuracy
- Precision
- Recall
- AUC Score

### Accuracy

All models achieved relatively high accuracy. However, because the dataset is imbalanced, accuracy can be misleading.

A model may predict most employees as non-attrition and still get high accuracy.

### Precision and Recall

Precision and recall are more useful for evaluating the minority class, which is employee attrition.

Recall is especially important in this project because it shows how well the model identifies employees who are actually likely to leave.

The Neural Network and Decision Tree performed better than KNN for identifying the attrition class.

### AUC Score

The Decision Tree achieved the highest AUC score among the tested models.

This means it was better at distinguishing between employees who may leave and employees who may stay.

## Result Summary

Based on the model comparison:

- KNN was simple but performed weaker for the minority class.
- Decision Tree gave the best overall performance.
- Neural Network also performed well, especially compared to KNN.
- Class imbalance remained the biggest challenge.

The Decision Tree model achieved the best AUC score, making it the most suitable model among the tested options.

## Conclusion

This project shows that employee attrition prediction is possible using machine learning, but class imbalance makes the task challenging.

The dataset had far more employees who stayed than employees who left. Because of this, the models struggled to correctly detect the minority class.

Among the tested models, the Decision Tree gave the best overall performance based on AUC score.

In the future, the performance could be improved by using better imbalance-handling techniques such as:

- SMOTE
- Class weighting
- More advanced ensemble models
- Hyperparameter tuning
- Collecting more attrition examples

Overall, this project provides a simple machine learning pipeline for predicting employee attrition and comparing multiple models.
