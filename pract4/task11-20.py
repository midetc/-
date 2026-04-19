import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import correlate
from scipy.integrate import odeint
from scipy.optimize import differential_evolution
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
import networkx as nx
from prophet import Prophet
import time
import os

def task_11():
    data = np.random.randn(100)
    data[50:] -= 5 
    plt.plot(data)
    plt.axvline(50, color='r', linestyle='--', label='Event')
    plt.legend()
    plt.show()

def task_12():
    df = pd.DataFrame({
        'ds': pd.date_range(start='2020-01-01', periods=100),
        'y': np.random.randn(100).cumsum()
    })
    m = Prophet()
    m.fit(df)
    future = m.make_future_dataframe(periods=10)
    forecast = m.predict(future)
    m.plot(forecast)
    plt.show()

def task_13():
    X = np.random.rand(100, 2)
    km = KMeans(n_clusters=3).fit(X)
    db = DBSCAN(eps=0.1).fit(X)
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.scatter(X[:, 0], X[:, 1], c=km.labels_)
    ax2.scatter(X[:, 0], X[:, 1], c=db.labels_)
    plt.show()

def task_14():
    solar = np.sin(np.linspace(0, 10, 100))
    energy = np.sin(np.linspace(0, 10, 100) - 0.5)
    corr = correlate(solar, energy)
    plt.plot(corr)
    plt.show()

def task_15():
    loads = np.random.rand(10)
    balanced = np.full(10, np.mean(loads))
    plt.bar(range(10), loads, alpha=0.5, label='Static')
    plt.plot(range(10), balanced, 'r-', label='Balanced')
    plt.legend()
    plt.show()

def task_16():
    X = np.random.rand(100, 5)
    pca = PCA(n_components=2).fit_transform(X)
    plt.scatter(pca[:, 0], pca[:, 1])
    plt.show()

def task_17():
    def cost_func(x):
        return np.sum(x**2)
    bounds = [(-10, 10), (-10, 10)]
    res = differential_evolution(cost_func, bounds)
    plt.scatter(res.x[0], res.x[1], c='r')
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    plt.show()

def task_18():
    G = nx.erdos_renyi_graph(10, 0.4)
    nx.draw(G, with_labels=True)
    plt.show()

def task_19():
    def model(z, t):
        D, S = z
        dDdt = 0.1 * S - 0.2 * D
        dSdt = 0.15 * D - 0.1 * S
        return [dDdt, dSdt]
    t = np.linspace(0, 10, 100)
    res = odeint(model, [100, 50], t)
    plt.plot(t, res[:, 0], label='Demand')
    plt.plot(t, res[:, 1], label='Supply')
    plt.legend()
    plt.show()

def task_20():
    df = pd.DataFrame(np.random.randn(10000, 10))
    times = {}
    
    t0 = time.time()
    df.to_csv('test.csv')
    times['CSV'] = time.time() - t0
    
    t0 = time.time()
    df.to_parquet('test.parquet')
    times['Parquet'] = time.time() - t0
    
    t0 = time.time()
    df.to_hdf('test.h5', key='df', mode='w')
    times['HDF5'] = time.time() - t0

    plt.bar(times.keys(), times.values())
    plt.ylabel('Time (s)')
    plt.show()
    
    for ext in ['csv', 'parquet', 'h5']:
        if os.path.exists(f'test.{ext}'):
            os.remove(f'test.{ext}')
