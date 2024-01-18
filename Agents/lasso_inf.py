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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from lifelines.utils import concordance_index as cindex

def lasso(test_data = []):
    set_config(display="text")  # displays text representation of estimators
    
    df = pd.read_pickle('./lasso_dataframe.pkl')
    # df = pd.read_csv('./TransPath/clinical.tsv', sep='\t')
    # df.replace("'--", np.nan, inplace=True)
    # df = df.dropna(axis=1, how='all')
    
    # test_data = ['TCGA-49-4501','TCGA-49-6742','TCGA-75-7030','TCGA-86-6562']
    if test_data:
    # df = df.head(10)
        df = df[df['case_submitter_id'].isin(test_data)]
    df = df.drop(columns=['case_submitter_id'])
    # categorical_variables = [ 'gender', 'race',
    #                       'ajcc_pathologic_stage','vital_status']
    # df_c = df[categorical_variables].astype('category')
    # categorical_variables = [ 'gender', 'race',
    #                       'ajcc_pathologic_stage','vital_status']
    
    # Xt = OneHotEncoder().fit_transform(df_c[categorical_variables])
    
    # df_final = pd.concat([Xt, df[['age_at_index', 'days_to_last_follow_up','days_to_death']]], axis=1)
    # df_2 = df_final
    # df_2.replace(np.nan, df_2.mode().iloc[0], inplace=True)
    
    # df_2['survival_status'] = df_2.apply(lambda row: [True, row['days_to_death']] if row['vital_status=Dead'] == 1 else [False, row['days_to_last_follow_up']], axis=1)
    # df_2['survival_status_lasso'] = df_2.apply(lambda row: row['days_to_death'] if row['vital_status=Dead'] == 1 else row['days_to_last_follow_up'], axis=1)
    
    
    
    
    # Assuming X is your feature matrix and 'status_tuple' is the structured array
    X = df.drop(columns=['survival_status','survival_status_lasso'])
    Y = df['survival_status_lasso']
    
    scaler = StandardScaler()
    X_test_scaled = scaler.fit_transform(X)
    
    lasso_model = joblib.load('lasso.joblib')
    predicted_survival_times = lasso_model.predict(X_test_scaled)
    # print(X['vital_status=Dead'], -predicted_survival_times, Y)
    # Calculating the C-index
    c_index = cindex(X['vital_status=Dead'], -predicted_survival_times, Y)
    # print(f"C-index: {c_index}")
    return c_index,predicted_survival_times