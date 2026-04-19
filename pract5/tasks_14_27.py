import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_selection import RFE
import networkx as nx
import multiprocessing
import cv2
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from textblob import TextBlob

def task_14():
    data = np.sin(np.linspace(0, 10, 100)) + np.random.normal(0, 0.1, 100)
    model = ARIMA(data, order=(5, 1, 0)).fit()
    forecast = model.forecast(steps=10)

def task_15():
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    world = world[(world.pop_est > 0) & (world.name != "Antarctica")]
    world['gdp_per_cap'] = world.gdp_md_est / world.pop_est
    world.plot(column='gdp_per_cap', cmap='OrRd', legend=True)
    plt.show()

def task_16():
    X = np.random.rand(300, 10)
    y = (X[:, 0] + X[:, 1] > 1).astype(int)
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X, y)

def task_17():
    ts = pd.Series([1.0, np.nan, np.nan, 4.0, 5.0, np.nan, 7.0])
    ts_linear = ts.interpolate(method='linear')
    ts_spline = ts.interpolate(method='spline', order=2)

def task_18():
    G = nx.barabasi_albert_graph(50, 2)
    nx.draw(G, node_size=50, with_labels=False)
    plt.show()

def process_chunk(chunk):
    return chunk ** 2

def task_19():
    data = np.arange(1000000)
    chunks = np.array_split(data, 4)
    with multiprocessing.Pool(processes=4) as pool:
        results = pool.map(process_chunk, chunks)
    processed_data = np.concatenate(results)

def task_20():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.circle(img, (50, 50), 30, (255, 255, 255), -1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)

def task_21():
    X = np.random.rand(500, 15)
    y = np.random.randint(0, 2, 500)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = LogisticRegression().fit(X_train, y_train)

def task_22():
    X = np.random.rand(1000, 20)
    y = np.random.randint(0, 2, 1000)
    model = Sequential([
        Dense(64, activation='relu', input_shape=(20,)),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X, y, epochs=5, batch_size=32, verbose=0)

def task_23():
    X = np.random.rand(200, 5)
    y = np.random.randint(0, 2, 200)
    param_grid = {'n_estimators': [10, 50, 100], 'max_depth': [None, 5, 10]}
    rf = RandomForestClassifier()
    grid_search = GridSearchCV(rf, param_grid, cv=3)
    grid_search.fit(X, y)

def task_24():
    X = np.random.rand(200, 10)
    y = 2 * X[:, 0] + 3 * X[:, 5] + np.random.randn(200)
    estimator = LinearRegression()
    selector = RFE(estimator, n_features_to_select=3, step=1)
    selector = selector.fit(X, y)

def task_25():
    tweets = ["I love this completely!", "This is the worst disaster.", "It is okay, nothing special."]
    sentiments = []
    for t in tweets:
        analysis = TextBlob(t)
        sentiments.append(analysis.sentiment.polarity)

def task_26():
    df = pd.DataFrame({'price': np.cumsum(np.random.randn(500) + 0.1)})
    df['lag_1'] = df['price'].shift(1)
    df = df.dropna()
    X = df[['lag_1']]
    y = df['price']
    model = LinearRegression().fit(X, y)

def task_27():
    X = pd.DataFrame({
        'volatility': np.random.rand(100),
        'debt_ratio': np.random.rand(100),
        'inflation': np.random.rand(100)
    })
    y = X['volatility'] * 0.4 + X['debt_ratio'] * 0.5 - X['inflation'] * 0.1 + np.random.randn(100)*0.1
    model = RandomForestRegressor().fit(X, y)
    importances = model.feature_importances_
