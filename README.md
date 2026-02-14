# Titanic Survival Prediction – ML Assignment 2

## a. Problem Statement
The objective is to build multiple classification models to predict passenger
survival on the Titanic and deploy them using a Streamlit web application.

## b. Dataset Description
The Titanic dataset contains 891 passenger records with demographic and travel
information. The target variable is Survived (binary classification).

## c. Model Comparison Table

| Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|------|----------|-----|----------|--------|----|-----|
| Logistic Regression | 0.810056 |0.861001 |0.796610 |0.681159 |0.734375 |0.592314 |
| Decision Tree | 0.793296         |0.776548 |0.735294 |0.724638 |0.729927 |0.562560 |
| KNN |   0.804469       |0.857642 |0.757576 |0.724638 | 0.740741| 0.584286|
| Naive Bayes |  0.793296        |0.845982 |0.716216 |0.768116 |0.741259 |0.570482 |
| Random Forest | 0.815642         |0.837154 |0.764706 |0.753623 |0.759124 |0.609858 |
| XGBoost |  0.826816        | 0.834651|0.796875 |0.739130 |0.766917 |0.630576 |


## Model Observations

| Model | Observation |
|-----|-------------|
| Logistic Regression | Interpretable baseline |
| Decision Tree | Overfits without pruning |
| KNN | Sensitive to scaling |
| Naive Bayes | Fast but assumption-heavy |
| Random Forest | Strong generalization |
| XGBoost | Best overall performance |