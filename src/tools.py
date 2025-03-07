import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import joblib
import psutil
import warnings
warnings.filterwarnings('ignore')

def data_overview(df, head=True, describe=True):
    '''数据详情'''
    # 基本信息
    if head:
        display(df.head())
    if describe:
        display(df.describe().T)
    # 缺失值
    missing_values = df.isnull().sum()
    missing_percentage = (df.isnull().sum() / len(df)) * 100
    overview = pd.DataFrame({
        'Missing Count': missing_values,
        'Missing(%)': missing_percentage.round(1)
    })
    # 各变量取值数量
    unique_counts = df.nunique()
    overview['Unique_counts'] = df.nunique()
    # 各变量取值范围
    unique_values = {col: df[col].unique() for col in df.columns}
    unique_value = []
    for col in unique_counts.index:
        if unique_counts[col] <= 20:
            unique_value.append(unique_values[col])
        else:
            unique_value.append('Much')
    overview['Unique_values'] = unique_value
    # 数据类型
    overview['Dtype'] = df.dtypes
    display(overview)
    # 重复值
    duplicate_count = df.duplicated().sum()
    print(f"Duplicate Rows Count: {duplicate_count} duplicate rows found" 
          if duplicate_count > 0 else "No duplicate rows found")

def plot_distribution(df, colunms: list, type: str):
    '''
    变量的数据分布可视化
    type:
        num: 数值型变量
        cat: 分类变量
    '''
    h = (len(colunms) - 1) // 3 + 1  # 判断需要多少行
    if len(colunms) == 1 and type == 'num':
        sns.histplot(df[colunms[0]], kde=True)
        h = -1
    if len(colunms) == 1 and type == 'cat':
        sns.countplot(df, x=colunms[0], palette="rocket")
        h = -1

    if type == 'num' and h == 1:
        fig, axes = plt.subplots(h, 3, figsize=(18, 6 * h))
        for i in range(len(colunms)):
            sns.histplot(df[colunms[i]], kde=True, ax=axes[i])
            
    if type == 'num' and h > 1:
        fig, axes = plt.subplots(h, 3, figsize=(18, 6 * h))
        for i in range(len(colunms)):
            sns.histplot(df[colunms[i]], kde=True, ax=axes[i//3, i%3])

    if type == 'cat' and h == 1:
        fig, axes = plt.subplots(h, 3, figsize=(18, 6 * h))
        for i in range(len(colunms)):
            sns.countplot(df, x=colunms[i], palette="rocket", ax=axes[i])
            
    if type == 'cat' and h > 1:
        fig, axes = plt.subplots(h, 3, figsize=(18, 6 * h))
        for i in range(len(colunms)):
            sns.countplot(df, x=colunms[i], palette="rocket", ax=axes[i//3, i%3])
            
    if type == 'num':
        plt.suptitle('Distribution of Numerical Features', fontsize=16, y=1.02)
    if type == 'cat':
        plt.suptitle('Distribution of Categorical Features', fontsize=16, y=1.02)
        
    plt.tight_layout()
    plt.show()

def plot_heat(df, num_columns, cat_columns):
    data_encoded = pd.DataFrame()
    # 分类变量使用标签编码
    label_encoder = LabelEncoder()
    for i in cat_columns:
        data_encoded[i] = label_encoder.fit_transform(df[i])
    # 数值型变量
    scaler = StandardScaler()
    data_encoded[num_columns] = scaler.fit_transform(df[num_columns])
    # 计算相关性矩阵
    heatmap_data = data_encoded.corr()
    # 绘制热图
    plt.figure(figsize=(10, 8))
    sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap='coolwarm', linewidths=0.5)
    plt.title('Heatmap of Categorical and Numerical Variables')
    plt.show()

def preprocess_num(df: pd.DataFrame, columns_num: list):
    '''
    连续变量的处理 (数据标准化)\n
    df: 包含需要处理的数值型数据\n
    columns_cat: 数值型变量的列名
    return: 包含处理前和处理后的数据框
    '''
    scaler = StandardScaler()
    df_standardized = pd.DataFrame(scaler.fit_transform(df[columns_num]), 
                                   columns=[item + 'Standard' for item in df[columns_num].columns])
    df = pd.concat([df, df_standardized], axis=1)
    print([item + 'Standard' for item in df[columns_num].columns])
    return df

def preprocess_cat(df: pd.DataFrame, columns_cat: list):
    '''
    分类变量的处理（独热编码）\n
    df: 需要处理的包含分类变量的数据\n
    columns_cat: 分类变量的列名\n
    return: 包含处理前和处理后的数据框
    '''
    encoder = OneHotEncoder(sparse=False)
    df_cat = pd.DataFrame(encoder.fit_transform(df[columns_cat]),
                         columns=encoder.get_feature_names_out(columns_cat)).astype('category')
    df = pd.concat([df, df_cat], axis=1)
    columns_cat_processed = encoder.get_feature_names_out(columns_cat)
    print(list(columns_cat_processed))
    return df


def impute_num(df, columns_X, columns_y, model):
    '''
    使用模型填充的方式填补**数值型**缺失值\n
    df: 含有缺失值的数据框\n
    columns_X: 填补缺失值时使用的特征X的列名，要求没有缺失值\n
    columns_y: 需要填补的列\n
    model: 填补时使用的sklearn中的模型，可选的模型包括`RandomForestRegressor`, `LinearRegression`, `SVR`\n
    return: 返回填充好的数据框
    '''
    df_X_y = df[columns_X + [columns_y]].dropna().reset_index(drop=True)
    # 划分训练集与测试集
    X_train, X_test, y_train, y_test = train_test_split(df_X_y[columns_X], df_X_y[columns_y], test_size=0.3, random_state=2025)
    # 训练模型
    model.fit(X_train, y_train)
    # 模型效果
    y_pred = model.predict(X_test)
    print(model)
    print(f'Mean Squared Error: {mean_squared_error(y_test, y_pred)}')
    if df[columns_y].isna().sum():
        df.loc[df[columns_y].isna(), columns_y] = model.predict(df[columns_X][df[columns_y].isna()])
    return df

def submit(model_name : str,
           y_name : str,
           test : pd.DataFrame):
    ''' 
    保存提交（预测）的数据\n
    model_name: 模型的名称（只传入点号之前的名称）\n
    test: 需要预测的数据集
    '''
    # 载入模型
    model = joblib.load(f'../models/{model_name}.pkl')

    # 使用模型预测
    y_pred = model.predict(test)

    # 保存提交
    submission = pd.read_csv('../submission/submission.csv')
    submission[y_name] = y_pred.astype(int)
    submission.to_csv(f'../submission/{model_name}.csv', index=None)


def save_model(model, name):
    '''保存模型'''
    joblib.dump(model, f'../models/{name}.pkl')
    print(f'{name} is successfully saved!')
    return True

def memory(df=None):
    '''
    查看内存使用情况
    df: 传入一个数据框，检测数据框的大小，若没有，则打印电脑的内存
    '''
    if not df.empty:
        print(f'内存使用：{df.memory_usage().sum() / 1024**2 : .2f} MB')
    else:
        mem = psutil.virtual_memory()
        print(f"可用内存: {mem.available / 1024 / 1024:.2f} MB")
        print(f"内存使用率: {mem.percent}%")
    

# memory()