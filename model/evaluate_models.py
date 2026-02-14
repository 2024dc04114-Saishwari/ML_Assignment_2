import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef
)

# =============================
# LOAD DATASET
# =============================
df = pd.read_csv("titanic.csv")

# =============================
# HANDLE MISSING VALUES (NO inplace)
# =============================
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# =============================
# FEATURE ENGINEERING
# =============================
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
df['Title'] = df['Title'].replace(
    ['Lady','Countess','Capt','Col','Don','Dr','Major','Rev','Sir','Jonkheer','Dona'],
    'Rare'
)
df['Title'] = df['Title'].replace(['Mlle','Ms'], 'Miss')
df['Title'] = df['Title'].replace('Mme', 'Mrs')

# =============================
# ENCODING
# =============================
label_cols = ['Sex', 'Embarked', 'Title']
le = LabelEncoder()
for col in label_cols:
    df[col] = le.fit_transform(df[col])

# =============================
# FEATURES & TARGET
# =============================
features = [
    'Pclass','Sex','Age','SibSp','Parch','Fare',
    'Embarked','FamilySize','IsAlone','Title'
]

X = df[features]
y = df['Survived']

# =============================
# TRAIN–TEST SPLIT
# =============================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =============================
# SCALING
# =============================
scaler = joblib.load("model/saved_models/scaler.pkl")
X_test = scaler.transform(X_test)

# =============================
# EVALUATE MODELS
# =============================
model_names = [
    "logistic",
    "decision_tree",
    "knn",
    "naive_bayes",
    "random_forest",
    "xgboost"
]

results = []

for name in model_names:
    model = joblib.load(f"model/saved_models/{name}.pkl")

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC": roc_auc_score(y_test, y_prob),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred),
        "MCC": matthews_corrcoef(y_test, y_pred)
    })

# =============================
# RESULTS TABLE
# =============================
results_df = pd.DataFrame(results)
print("\nModel Evaluation Results:\n")
print(results_df)
