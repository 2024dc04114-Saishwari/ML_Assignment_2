import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Load dataset
df = pd.read_csv("titanic.csv")

# -------------------------
# Feature Engineering
# -------------------------
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])


df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
df['Title'] = df['Title'].replace(
    ['Lady', 'Countess','Capt', 'Col','Don','Dr','Major','Rev','Sir','Jonkheer','Dona'],
    'Rare'
)
df['Title'] = df['Title'].replace(['Mlle', 'Ms'], 'Miss')
df['Title'] = df['Title'].replace('Mme', 'Mrs')

# Encoding
label_cols = ['Sex', 'Embarked', 'Title']
le = LabelEncoder()
for col in label_cols:
    df[col] = le.fit_transform(df[col])

# Features and target
features = [
    'Pclass','Sex','Age','SibSp','Parch','Fare',
    'Embarked','FamilySize','IsAlone','Title'
]

X = df[features]
y = df['Survived']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

joblib.dump(scaler, "model/saved_models/scaler.pkl")

# Models
models = {
    "logistic": LogisticRegression(max_iter=1000),
    "decision_tree": DecisionTreeClassifier(),
    "knn": KNeighborsClassifier(n_neighbors=7),
    "naive_bayes": GaussianNB(),
    "random_forest": RandomForestClassifier(n_estimators=200),
    "xgboost": XGBClassifier(eval_metric="logloss")
}

# Train & save models
for name, model in models.items():
    model.fit(X_train, y_train)
    joblib.dump(model, f"model/saved_models/{name}.pkl")

print("All Titanic models trained successfully.")
