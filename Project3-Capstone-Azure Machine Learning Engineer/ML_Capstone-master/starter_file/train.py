
# Import libraries
from sklearn.linear_model import LogisticRegression
import argparse
import os
import numpy as np
from sklearn.metrics import mean_squared_error
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from azureml.core.run import Run
from azureml.data.dataset_factory import TabularDatasetFactory
from azureml.core import Dataset
from azureml.data.datapath import DataPath
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

# Set regularization parameter
parser = argparse.ArgumentParser()

parser.add_argument('--C', type=float, default=1.0, help="Inverse of regularization strength. Smaller values cause stronger regularization")
parser.add_argument('--max_iter', type=int, default=100, help="Maximum number of iterations to converge")
parser.add_argument('--regularization', type=float, dest='reg_rate', default=0.01, help='regularization rate')
args = parser.parse_args()
reg = args.reg_rate

# Get the experiment run context
run = Run.get_context()

# load the heart_failure_clinical_records_dataset
ds= TabularDatasetFactory.from_delimited_files(path="https://archive.ics.uci.edu/ml/machine-learning-databases/00519/heart_failure_clinical_records_dataset.csv")
def clean_data(data):
    
    x_df = data.to_pandas_dataframe().dropna()
    y_df = x_df.pop("DEATH_EVENT")
    return x_df, y_df

x, y = clean_data(ds)

# Separate features and labels
#X, y = ds['age','anaemia','creatinine_phosphokinase','diabetes','ejection_fraction','high_blood_pressure','platelets',
  #               'serum_creatinine','serum_sodium','sex','smoking','time'].values, ds['DEATH_EVENT'].values

# Split data into training set and test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=0)

# Train a logistic regression model
print('Training a logistic regression model with regularization rate of', reg)
run.log('Regularization Rate',  np.float(reg))
run.log("Regularization Strength:", np.float(args.C))
run.log("Max iterations:", np.int(args.max_iter))
model = LogisticRegression(C=args.C,max_iter=args.max_iter, solver="liblinear").fit(x_train, y_train)

# calculate accuracy
y_sc = model.predict(x_test)
acc = np.average(y_sc == y_test)
print('Accuracy:', acc)
run.log('Accuracy', np.float(acc))

# calculate AUC
y_scores = model.predict_proba(x_test)
auc = roc_auc_score(y_test,y_scores[:,1])
print('AUC: ' + str(auc))
run.log('AUC', np.float(auc))

os.makedirs('outputs', exist_ok=True)
# note file saved in the outputs folder is automatically uploaded into experiment record
joblib.dump(value=model, filename='outputs/best_run_hd.pkl')

run.complete()
