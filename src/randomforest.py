from tools import *
from models import *
import pandas as pd

train = pd.read_csv('input/train.csv')
test = pd.read_csv('input/test.csv')

# 合并训练集与测试集
train["DatasetType"] = "train"
test["DatasetType"] = "test"
combined = pd.concat([train, test], axis=0).reset_index(drop=True)

# 特征类型
num_columns = ['Compartments', 'Weight Capacity (kg)']
cat_columns = ['Brand', 'Material', 'Size', 'Laptop Compartment', 'Waterproof', 'Style', 'Color']
feature_columns = ['Price']

# 数值型数据归一化，离散型数据独热编码
train = preprocess_cat(train, cat_columns).drop(cat_columns, axis=1)
train = preprocess_num(train, num_columns).drop(num_columns, axis=1)
train['Weight Capacity (kg)Standard'] = train['Weight Capacity (kg)Standard'].fillna(train['Weight Capacity (kg)Standard'].mean())
X = train.drop(['id', 'Price', 'DatasetType'], axis=1)
y = train['Price']

test = preprocess_cat(test, cat_columns).drop(cat_columns, axis=1)
test = preprocess_num(test, num_columns).drop(num_columns, axis=1)
test['Weight Capacity (kg)Standard'] = test['Weight Capacity (kg)Standard'].fillna(test['Weight Capacity (kg)Standard'].mean())
test = test.drop(['id', 'DatasetType'], axis=1)

# 划分训练集与测试集
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

lightgbm = best_lightgbm(X, y, scoring='neg_mean_absolute_error', task='reg')
save_model(lightgbm, 'lightgbm')
submit('lightgbm', test)
