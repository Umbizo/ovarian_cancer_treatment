"""
This algorithm cleans the METABRIC dataset and seperates it into an X table of predictive features, and two y tables.
y_mortality is for mortality prediction
y_survival is for risk prediction
"""

import pandas as pd
import numpy as np

#Original Metabric Dataset
df = pd.read_csv('archive.zip')

#print(df.isna().sum())
#Around the last 500 rows contain several null values, so we're removing them.
df_clean = df.iloc[:1986]

#Columns removed due to redundancy and/or several null values
df_clean = df_clean.drop(columns=["3-Gene classifier subtype", "Tumor Stage"])

#Dropping null rows where important features are null
df_clean = df_clean.dropna(subset=['Chemotherapy'])
df_clean = df_clean.dropna(subset=['Relapse Free Status'])

#Columns removed due to redundancy
df_clean = df_clean.drop(columns=["ER status measured by IHC", "HER2 status measured by SNP6"])

#Filling empty values from column to "Unspecified"
df_clean['Primary Tumor Laterality'] = df_clean['Primary Tumor Laterality'].fillna('Unspecified')

#Creation for the ICD10 column in order to fit modern diagnosis classification.
#This column is derived from Primary Tumor Laterality and Sex
sex_map = {"Female": "1", "Male": "2"} #Mapping gender to numbers
lat_map = {"Right": "1", "Left": "2", "Unspecified": "9"} #Mapping laterality to numbers
df_clean ["ICD10"] = ("C50.9"
                  + df_clean["Sex"].map(sex_map)
                  + df_clean["Primary Tumor Laterality"].map(lat_map))


#df_clean.to_csv('df_clean.csv', index=False)
#X = pd.read_csv('df_clean.csv')
X = df_clean

#Creating the y table for mortality
y_mortality = X[["Overall Survival Status","Overall Survival (Months)", "Patient's Vital Status"]]
X = X.drop(columns=["Overall Survival Status","Overall Survival (Months)", "Patient's Vital Status"])

#Creating the y table for survival
y_relapse = X[["Relapse Free Status", "Relapse Free Status (Months)"]]
X = X.drop(columns=["Relapse Free Status", "Relapse Free Status (Months)"])

#More columns removed due to redundancy or because they're classifiers:
X = X.drop(columns=["Inferred Menopausal State", "Nottingham prognostic index", "Sex", "Integrative Cluster", "Primary Tumor Laterality"])
X = X.drop(columns=["Patient ID", "Cancer Type", "Oncotree Code"], errors="ignore")

#X table completed.
#X.to_csv('X.csv', index=False)
