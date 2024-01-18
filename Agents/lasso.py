import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import joblib
from sksurv.datasets import load_breast_cancer
from sksurv.linear_model import CoxPHSurvivalAnalysis, CoxnetSurvivalAnalysis
from sksurv.preprocessing import OneHotEncoder

from sklearn import set_config
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

set_config(display="text")  # displays text representation of estimators

df = pd.read_csv('./TransPath/clinical.tsv', sep='\t')
df.replace("'--", np.nan, inplace=True)
df = df.dropna(axis=1, how='all')
case_id = df['case_submitter_id']
categorical_variables = [ 'gender', 'race',
                       'ajcc_pathologic_stage','vital_status']
df_c = df[categorical_variables].astype('category')
categorical_variables = [ 'gender', 'race',
                       'ajcc_pathologic_stage','vital_status']

Xt = OneHotEncoder().fit_transform(df_c[categorical_variables])

df_final = pd.concat([Xt, df[['age_at_index', 'days_to_last_follow_up','days_to_death']]], axis=1)
df_2 = df_final
df_2.replace(np.nan, df_2.mode().iloc[0], inplace=True)

df_2['survival_status'] = df_2.apply(lambda row: [True, row['days_to_death']] if row['vital_status=Dead'] == 1 else [False, row['days_to_last_follow_up']], axis=1)
df_2['survival_status_lasso'] = df_2.apply(lambda row: row['days_to_death'] if row['vital_status=Dead'] == 1 else row['days_to_last_follow_up'], axis=1)

# df_3 = df_2
df_2['case_submitter_id'] = case_id
df_2.to_pickle('lasso_dataframe.pkl')
df_2 = df_2.drop(columns=['case_submitter_id'])

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from lifelines.utils import concordance_index as cindex

# Assuming X is your feature matrix and 'status_tuple' is the structured array
X = df_2.drop(columns=['survival_status','survival_status_lasso'])
Y = df_2['survival_status_lasso']

# Splitting the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Standardizing the features (optional but recommended for lasso models)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Fitting the lasso regression model
lasso_model = Lasso(alpha=0.01)  # Adjust alpha as needed
lasso_model.fit(X_train_scaled, Y_train)

# Predicting the survival times for the test set (replace with appropriate prediction method)
joblib.dump(lasso_model, "lasso.joblib")
predicted_survival_times = lasso_model.predict(X_test_scaled)

# Calculating the C-index
c_index = cindex(X_test['vital_status=Dead'], -predicted_survival_times, Y_test)
print(f"C-index: {c_index}")
