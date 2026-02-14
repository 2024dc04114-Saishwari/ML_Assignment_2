import streamlit as st
import pandas as pd
import joblib

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

st.title("Titanic Survival Prediction App")

# =============================
# MODEL SELECTION
# =============================
model_name = st.selectbox(
    "Select Model",
    ["logistic", "decision_tree", "knn",
     "naive_bayes", "random_forest", "xgboost"]
)

# =============================
# LOAD MODEL & SCALER
# =============================
model = joblib.load(f"model/saved_models/{model_name}.pkl")
scaler = joblib.load("model/saved_models/scaler.pkl")

# =============================
# LOAD DATASET (FOR METRICS)
# =============================
df = pd.read_csv("titanic.csv")

# Preprocessing (same as training)
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
df['Title'] = df['Title'].replace(
    ['Lady','Countess','Capt','Col','Don','Dr','Major','Rev','Sir','Jonkheer','Dona'],
    'Rare'
)
df['Title'] = df['Title'].replace(['Mlle','Ms'], 'Miss')
df['Title'] = df['Title'].replace('Mme', 'Mrs')

label_cols = ['Sex', 'Embarked', 'Title']
le = LabelEncoder()
for col in label_cols:
    df[col] = le.fit_transform(df[col])

features = [
    'Pclass','Sex','Age','SibSp','Parch','Fare',
    'Embarked','FamilySize','IsAlone','Title'
]

X = df[features]
y = df['Survived']

# Train-test split for evaluation display
_, X_test, _, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_test_scaled = scaler.transform(X_test)
y_pred = model.predict(X_test_scaled)

# =============================
# DISPLAY METRICS
# =============================
st.subheader("Model Evaluation (Test Set)")

st.text("Classification Report")
st.text(classification_report(y_test, y_pred))

st.subheader("Confusion Matrix")
st.write(confusion_matrix(y_test, y_pred))

# =============================
# FILE UPLOAD FOR PREDICTION
# =============================
st.subheader("Upload CSV for Prediction")

uploaded_file = st.file_uploader(
    "Upload Titanic CSV (without 'Survived' column)",
    type=["csv"]
)

if uploaded_file:
    df_upload = pd.read_csv(uploaded_file)

    # Preprocess uploaded data
    df_upload['Age'] = df_upload['Age'].fillna(df_upload['Age'].median())
    df_upload['Embarked'] = df_upload['Embarked'].fillna(df_upload['Embarked'].mode()[0])

    df_upload['FamilySize'] = df_upload['SibSp'] + df_upload['Parch'] + 1
    df_upload['IsAlone'] = (df_upload['FamilySize'] == 1).astype(int)

    df_upload['Title'] = df_upload['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    df_upload['Title'] = df_upload['Title'].replace(
        ['Lady','Countess','Capt','Col','Don','Dr','Major','Rev','Sir','Jonkheer','Dona'],
        'Rare'
    )
    df_upload['Title'] = df_upload['Title'].replace(['Mlle','Ms'], 'Miss')
    df_upload['Title'] = df_upload['Title'].replace('Mme', 'Mrs')

    for col in label_cols:
        df_upload[col] = le.fit_transform(df_upload[col])

    X_upload = df_upload[features]
    X_upload_scaled = scaler.transform(X_upload)

    predictions = model.predict(X_upload_scaled)
    probabilities = model.predict_proba(X_upload_scaled)[:, 1]

    st.subheader("🔮 Prediction Results")
    df_upload['Survived_Prediction'] = predictions
    df_upload['Survival_Probability'] = probabilities

    st.write(df_upload[['Survived_Prediction', 'Survival_Probability']])
