"""
This algorithm creates a Machine Learning model to predict the risk of death of a patient diagnosed with Breast Cancer.
Categorical Variables are one-hot encoded and missing values are imputed.
The model used is Random Forest.
The target variables are "Overall Survival Status" and "Overall Survival Status (Months)", the model marks 1 as Deceased, 0 as Living.
For patients who are not deceased, the model doesn't consider them "alive forever". It just tells the model that this patient has 
survived for a long time.
Almost every line contains an explanation.
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv


X = pd.read_csv('X.csv')
y_df = pd.read_csv('y_mortality.csv')

#This ensures that patients who are living aren't considered as "living forever" for risk prediction
y_surv = Surv.from_arrays(event=(y_df["Overall Survival Status"] == "Deceased"),
                          time=y_df["Overall Survival (Months)"])

#Seperating numerical and categorical columns
num_cols = X.select_dtypes(include=np.number).columns.tolist()
cat_cols = [c for c in X.columns if c not in num_cols]

#One hot encoding and imputing
preprocess = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), num_cols),
    ("cat", Pipeline([
        ("imp", SimpleImputer(strategy="constant", fill_value="Missing")),
        ("ohe", OneHotEncoder(handle_unknown="ignore")),
    ]), cat_cols),
])

#Creating the RF model, parameters haven't been evaluated by me yet
model = Pipeline([
    ("prep", preprocess),
    ("rf", RandomSurvivalForest(n_estimators=300, min_samples_leaf=10, n_jobs=-1)),
])

import joblib
from sksurv.metrics import concordance_index_censored
from sklearn.inspection import permutation_importance

#Splitting
X_train, X_test, y_train_surv, y_test_surv = train_test_split(
    X, y_surv, test_size=0.2, stratify=y_surv["event"], random_state=42
)


model.fit(X_train, y_train_surv)

#Evaluating preformance using C-Index
risk = model.predict(X_test) #higher number means higher risk
cindex = concordance_index_censored(
    y_test_surv["event"], y_test_surv["time"], risk
)[0]
print("Test C-index:", cindex) #0.5 = random, 1.0 = perfect

#Deriving Feature importance using permutation
#Mapping each transformed column (one hot encoded) back to its original feature
perm = permutation_importance(
    model, X_test, y_test_surv, n_repeats=10, random_state=42, n_jobs=-1
)
imp = pd.Series(perm.importances_mean, index=X_test.columns)
grouped = imp.sort_values(ascending=False)

#Although the sums of the weights is already arond 1, doing this guarantees it.
weights = grouped / grouped.sum()

#Dataframe conversion
weights_df = pd.DataFrame({
    "weight": weights,
    "percent": (weights * 100).round(2),
})
print(weights_df)

#Exporting weights in case
#weights_df.to_csv('survival_feature_weights.csv')

#Exporting model
#joblib.dump(model, "rsf_breast_cancer_os.joblib")
