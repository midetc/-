import pandas as pd
import numpy as np
import dask.dataframe as dd
import time
import sqlite3
import json
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from faker import Faker
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import Counter
import matplotlib.pyplot as plt

def task_1():
    np.random.seed(42)
    df = pd.DataFrame({'value': np.random.normal(50, 10, 1000)})
    df.loc[10, 'value'] = 150
    df.loc[500, 'value'] = -50
    Q1 = df['value'].quantile(0.25)
    Q3 = df['value'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    anomalies = df[(df['value'] < lower_bound) | (df['value'] > upper_bound)]
    print(anomalies)

def task_2():
    df = pd.DataFrame({'A': np.random.rand(2000000), 'B': np.random.rand(2000000)})
    df.to_csv('big_data.csv', index=False)
    
    t0 = time.time()
    pd_df = pd.read_csv('big_data.csv')
    pd_mean = pd_df['A'].mean()
    print("Pandas time:", time.time() - t0)
    
    t0 = time.time()
    ddf = dd.read_csv('big_data.csv')
    dd_mean = ddf['A'].mean().compute()
    print("Dask time:", time.time() - t0)

def task_3():
    df = pd.DataFrame({'A': [10, 20, 30, 40, 50], 'B': [1, 2, 3, 4, 5]})
    scaler_std = StandardScaler()
    scaler_minmax = MinMaxScaler()
    df_std = pd.DataFrame(scaler_std.fit_transform(df), columns=df.columns)
    df_minmax = pd.DataFrame(scaler_minmax.fit_transform(df), columns=df.columns)

def task_4():
    pd.DataFrame({'id': [1, 2], 'val1': ['A', 'B']}).to_csv('t4.csv', index=False)
    with open('t4.json', 'w') as f: json.dump([{'id': 1, 'val2': 'X'}, {'id': 2, 'val2': 'Y'}], f)
    conn = sqlite3.connect('t4.db')
    pd.DataFrame({'id': [1, 2], 'val3': [100, 200]}).to_sql('t4_table', conn, index=False, if_exists='replace')
    
    df_csv = pd.read_csv('t4.csv')
    df_json = pd.read_json('t4.json')
    df_sql = pd.read_sql('SELECT * FROM t4_table', conn)
    
    merged = df_csv.merge(df_json, on='id').merge(df_sql, on='id')

def task_5():
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    text = "The quick brown foxes are jumping over the lazy dogs."
    words = text.lower().replace('.', '').split()
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    processed = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    freq = Counter(processed)

def task_6():
    X = np.random.rand(10000, 5)
    y = 3 * X[:, 0] + 1.5 * X[:, 2] + np.random.randn(10000)
    model = LinearRegression().fit(X, y)

def task_7():
    texts = ["I love this", "This is bad", "Great product", "Terrible service"]
    labels = [1, 0, 1, 0]
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(texts)
    nb_model = MultinomialNB().fit(X, labels)
    svm_model = SVC(kernel='linear').fit(X, labels)

def task_8():
    fake = Faker()
    data = [{'name': fake.name(), 'email': fake.email(), 'address': fake.address()} for _ in range(100)]
    df = pd.DataFrame(data)

def task_9():
    df = pd.DataFrame({'color': ['red', 'blue', 'green', 'red'], 'size': ['S', 'M', 'L', 'S']})
    df_onehot = pd.get_dummies(df, columns=['color'])
    le = LabelEncoder()
    df['size_encoded'] = le.fit_transform(df['size'])

def task_10():
    df = pd.DataFrame({'A': [1, np.nan, 3, 4], 'B': [np.nan, 2, 3, 4], 'C': [1, 2, np.nan, np.nan]})
    df['A_mean'] = SimpleImputer(strategy='mean').fit_transform(df[['A']])
    df['B_median'] = SimpleImputer(strategy='median').fit_transform(df[['B']])
    df['C_ffill'] = df['C'].ffill()
    df['C_bfill'] = df['C'].bfill()

def task_11():
    X = np.random.rand(200, 2)
    kmeans = KMeans(n_clusters=4, random_state=42).fit(X)
    plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_)
    plt.show()

def task_12():
    X = np.random.randn(500, 2)
    X[0] = [10, 10]
    iso = IsolationForest(contamination=0.01).fit(X)
    preds = iso.predict(X)

def task_13():
    X = np.random.rand(500, 4)
    y = X[:, 0] * 100 + X[:, 1] * 50 + np.random.randn(500) * 10
    model = RandomForestRegressor(n_estimators=50)
    scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
