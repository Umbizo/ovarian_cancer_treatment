"""
This algorithm creates a Machine Learning model to predict the mortality of a patient diagnosed with Breast Cancer.
Categorical Variables are one-hot encoded and missing values are imputed.
The model used is Random Forest.
The target variable is "Overall Survival Status", the model marks 1 as Deceased, 0 as Living.
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



X = pd.read_csv('X.csv')
y_df = pd.read_csv('y_mortality.csv')

#1 = Deceased, 0 = Living
y = (y_df["Overall Survival Status"] == "Deceased").astype(int)

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

#Creating the RF Model
#The parameters were chosen specifically based on another python file I used to find the parameters that yielded the best results
model = Pipeline([("prep", preprocess), 
                  ("rf", RandomForestClassifier(
                      n_estimators=1000,
                      min_samples_leaf=10,
                      class_weight="balanced",
                      n_jobs=-1,
                      random_state=42,
                      max_features=0.33
                      )),
])

#Splitting
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

#Cross validation AUC average on training set
print("CV AUC:", cross_val_score(model, X_train, y_train, cv=5, scoring="roc_auc").mean())

#Testing the model and getting AUC score on test set
model.fit(X_train, y_train)
proba = model.predict_proba(X_test)[:, 1]
print("Test AUC:", roc_auc_score(y_test, proba))

#Detailed classification report with precision, recall, f1, and support as well as accuracy, macro and weighted avg.
print(classification_report(y_test, model.predict(X_test)))


#Deriving Feature importance
names = model.named_steps["prep"].get_feature_names_out()
imp = pd.Series(model.named_steps["rf"].feature_importances_, index=names)
#print(imp.sort_values(ascending=False).head(15))

#Mapping each transformed column (one hot encoded) back to its original feature
def original_feature(name):
    if name.startswith("num__"):
        return name[len("num__"):]
    stripped = name[len("cat__"):]
    #The longest matching categorical column name wins
    matches = [c for c in cat_cols if stripped.startswith(c + "_")]
    return max(matches, key=len)

grouped = imp.groupby(original_feature).sum().sort_values(ascending=False)

#Although the sums of the weights is already arond 1, doing this guarantees it.
weights = grouped / grouped.sum()

#Dataframe conversion
weights_df = pd.DataFrame({
    "weight": weights,
    "percent": (weights * 100).round(2),
})
print(weights_df)

#Exporting weights in case
#weights_df.to_csv('mortality_feature_weights.csv')

#Exporting
#model.fit(X, y)  # optional: refit on all data before shipping
#joblib.dump(model, "breast_cancer_mortality_rf.joblib")

#Reloading the model and refitting it
#loaded = joblib.load("rf_breast_cancer_os.joblib")
#loaded.predict_proba(new_df)[:, 1]   # new_df needs the same columns as X
