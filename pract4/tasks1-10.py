import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import minimize
from scipy.interpolate import interp1d, UnivariateSpline
from sklearn.linear_model import BayesianRidge, LinearRegression
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans, DBSCAN
from statsmodels.tsa.seasonal import seasonal_decompose
from hmmlearn import hmm
import pywt
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Input

def task_1():
    def model(y, t, k):
        return -k * y
    t = np.linspace(0, 20, 100)
    y1 = odeint(model, 100, t, args=(0.1,))
    y2 = np.full_like(t, 50)
    plt.plot(t, y1, label='ODE')
    plt.plot(t, y2, label='Static Mean')
    plt.legend()
    plt.show()

def task_2():
    data = np.random.normal(0, 1, 100)
    data[50:] += 5
    cusum = np.cumsum(data - np.mean(data))
    plt.plot(data, label='Data')
    plt.plot(cusum, label='CUSUM')
    plt.legend()
    plt.show()

def task_3():
    X = np.random.rand(100, 1, 1)
    y = np.random.rand(100, 1)
    model = Sequential([LSTM(10, input_shape=(1, 1)), Dense(1)])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=2, verbose=0)
    pred = model.predict(X)
    plt.plot(y, label='True')
    plt.plot(pred, label='LSTM')
    plt.legend()
    plt.show()

def task_4():
    X = np.random.rand(100, 1)
    y = 3 * X.flatten() + np.random.randn(100)
    br = BayesianRidge().fit(X, y)
    lr = LinearRegression().fit(X, y)
    plt.scatter(X, y)
    plt.plot(X, br.predict(X), label='Bayesian')
    plt.plot(X, lr.predict(X), label='Linear')
    plt.legend()
    plt.show()

def task_5():
    def cost(x):
        return x[0]**2 + x[1]**2 + x[0]*x[1]
    res = minimize(cost, [10, 10])
    plt.bar(['Time 1', 'Time 2'], res.x)
    plt.show()

def task_6():
    X = np.random.randn(100, 1)
    X[20] = 10
    iso = IsolationForest(contamination=0.05).fit(X)
    preds = iso.predict(X)
    plt.scatter(range(100), X, c=preds)
    plt.show()

def task_7():
    X = np.random.rand(100, 10)
    input_layer = Input(shape=(10,))
    encoded = Dense(5, activation='relu')(input_layer)
    decoded = Dense(10, activation='sigmoid')(encoded)
    autoencoder = Model(input_layer, decoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    autoencoder.fit(X, X, epochs=2, verbose=0)
    reconstructed = autoencoder.predict(X)
    plt.plot(X[0], label='Original')
    plt.plot(reconstructed[0], label='Reconstructed')
    plt.legend()
    plt.show()

def task_8():
    X = np.random.randn(100, 1)
    model = hmm.GaussianHMM(n_components=2).fit(X)
    states = model.predict(X)
    plt.scatter(range(100), X, c=states)
    plt.show()

def task_9():
    data = np.sin(np.linspace(0, 20, 100)) + np.random.randn(100) * 0.1
    coeffs = pywt.wavedec(data, 'db1', level=2)
    plt.plot(data, label='Original')
    plt.plot(coeffs[0], label='Wavelet Approx')
    plt.legend()
    plt.show()

def task_10():
    x = np.linspace(0, 10, 10)
    y = np.sin(x)
    x_new = np.linspace(0, 10, 50)
    f_lin = interp1d(x, y, kind='linear')
    f_cub = interp1d(x, y, kind='cubic')
    plt.plot(x, y, 'o')
    plt.plot(x_new, f_lin(x_new), label='Linear')
    plt.plot(x_new, f_cub(x_new), label='Cubic')
    plt.legend()
    plt.show()
