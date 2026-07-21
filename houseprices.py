import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Use the kagglehub client library to attach Kaggle resources like competitions, datasets, and models to your session
# Learn more about kagglehub: https://github.com/Kaggle/kagglehub/blob/main/README.md

import kagglehub

from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from scipy.stats import skew
#Models
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, Lasso
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import GradientBoostingRegressor
from catboost import CatBoostRegressor
import lightgbm as lgb
import xgboost as xg

import warnings
warnings.filterwarnings("ignore")

data = pd.read_csv('../input/competitions/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv("../input/competitions/house-prices-advanced-regression-techniques/test.csv")

data_b = data.drop(["Id"], axis = 1)
test_b = test.drop(["Id"], axis = 1)
data_b.head()

data_b.isnull().sum().sort_values(ascending = False).head(20)

#Now we need to determine which columns has NA
null_in_train = data_b.columns[data_b.isna().any()].tolist()
null_ony_in_test = list(set(test_b.columns[test_b.isna().any()].tolist())-set(data_b.columns[data_b.isna().any()].tolist()))
all_null_cols = null_in_train + null_ony_in_test

print("/nNull in Train: " + str(null_in_train))
print("Null values only in Test: " + str(null_ony_in_test))
print("/nAll Null Cols: " + str(all_null_cols))

data_b = data_b.drop(all_null_cols, axis = "columns")
test_b = test_b.drop(all_null_cols, axis = "columns")

y = data_b.pop("SalePrice")

#Now we split columns as categorical and numerical columns
cols = data_b.columns
numeric_columns, categorical_columns = [], []
for i in range(len(cols)):
    if data_b[cols[i]].dtypes == 'O':
        categorical_columns.append(cols[i])
    else:
        numeric_columns.append(cols[i])
        
#To show categorical columns and its unique items
category_analysis = pd.DataFrame(categorical_columns, columns = ["Feature"])
unique_values = []
unique_counts = []
for col in categorical_columns:
    unique_values.append(data_b[col].unique())
    unique_counts.append(len(data_b[col].unique()))

category_analysis["Categories"] = unique_values
category_analysis["Number"] = unique_counts
category_analysis

#Lets make one hot encodings via pd.get_dummies() function
data_b = pd.get_dummies(data_b)
test_b = pd.get_dummies(test_b)

#Skewing all numeric features
skewed_feats = data_b[numeric_columns].apply(lambda x : skew(x.dropna())).sort_values(ascending = False)
print("\nSkew in numerical features: \n")
skewness = pd.DataFrame({'Skew' : skewed_feats})
skewness.head(10)

data_b_skewed = data_b.copy()
test_b_skewed = test_b.copy()
skewness = skewness[abs(skewness) > 1]
print("There are {} skewed numerical features to Log1p transform".format(skewness.shape[0]))

skewed_features = skewness.index
for feat in skewed_features:
    data_b_skewed[feat] = np.log1p(data_b_skewed[feat])
    test_b_skewed[feat] = np.log1p(test_b_skewed[feat])

#Also we need to convert target values
y_log = np.log1p(y)

print("\nOnly in Train: "+ str(list(set(data_b_skewed.columns) - set(test_b_skewed.columns))))
print("Only in Test: "+ str(list(set(test_b_skewed.columns) - set(data_b_skewed.columns))))

only_in_train = list(set(data_b_skewed.columns) - set(test_b_skewed.columns))
data_b_skewed = data_b_skewed.drop(only_in_train, axis='columns')

#Now we try to simulate each model by using cross validation
def run_cvs(X,y):
    baseline = ElasticNet(random_state = 0, max_iter = 10000000, alpha = 0.0003)
    baseline_score = cross_val_score(baseline, X, y, cv = 10)
    print("ENet avg: ", np.mean(baseline_score))

    baseline = Ridge(alpha = 1, random_state=0)
    baseline_score = cross_val_score(baseline, X, y, cv=10)
    print("Ridge avg:",np.mean(baseline_score))   
    
    baseline = Lasso(alpha = 0.0001,random_state=0)
    baseline_score = cross_val_score(baseline, X, y, cv=10)
    print("Lasso avg:",np.mean(baseline_score))
    
    baseline = KernelRidge(alpha=0.1)
    baseline_score = cross_val_score(baseline, X, y, cv=10)
    print("KRR avg:",np.mean(baseline_score))

    baseline = lgb.LGBMRegressor(learning_rate=0.01,num_leaves=4,n_estimators=2000, random_state=0)
    baseline_score = cross_val_score(baseline, X, y, cv=10)
    print("LGBM avg:",np.mean(baseline_score))

    baseline = xg.XGBRegressor(learning_rate=0.01,n_estimators=2000, subsample=0.7,colsample_bytree=0.7,random_state=0)
    baseline_score = cross_val_score(baseline, X, y, cv=10)
    print("XGB avg:",np.mean(baseline_score))
    
    baseline = CatBoostRegressor(random_state=0,verbose=0)
    baseline_score = cross_val_score(baseline, X, y, cv=10)
    print("CatB avg:",np.mean(baseline_score))

    baseline = GradientBoostingRegressor(n_estimators=1000, learning_rate=0.02,max_depth=4, max_features='sqrt', min_samples_leaf=15, 
               min_samples_split=50,loss='huber', random_state = 0)
    baseline_score = cross_val_score(baseline, X, y, cv=10)
    print("GBR avg:",np.mean(baseline_score))
    
run_cvs(data_b_skewed, y_log)

#Combine 4 models (2 tree-based AND 2 linear Models) 
def make_submission(X_train, y_train, X_test):    
    sub_df = pd.read_csv('/kaggle/input/competitions/house-prices-advanced-regression-techniques/sample_submission.csv', index_col = "Id")
    
    ridge = Ridge(alpha = 1, random_state=0).fit(X_train,y_train)
    ridge_preds_log=ridge.predict(X_test)
    
    lasso = Lasso(alpha = 0.0001,random_state=0).fit(X_train,y_train)
    lasso_preds_log=lasso.predict(X_test)

    catB = CatBoostRegressor(random_state=0,verbose=0).fit(X_train,y_train)
    catB_preds_log=catB.predict(X_test)

    xgb = xg.XGBRegressor(learning_rate=0.01,n_estimators=2000, subsample=0.7,colsample_bytree=0.7,random_state=0).fit(X_train,y_train)
    xgb_preds_log=xgb.predict(X_test)
    
    catb_xbr_lasso_ridge_mean_preds_log=(catB_preds_log+ridge_preds_log+lasso_preds_log+xgb_preds_log)/4
    sub_df['SalePrice'] = np.exp(catb_xbr_lasso_ridge_mean_preds_log)-1
    sub_df.to_csv("submission.csv")
    
make_submission(data_b_skewed,y_log,test_b_skewed)
