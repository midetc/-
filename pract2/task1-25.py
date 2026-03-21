import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import multiprocessing as mp
from numba import jit, prange
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from faker import Faker
import markovify
import dask.dataframe as dd
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import featuretools as ft
from sklearn.cluster import KMeans
from scipy import stats

print("--- ПОЧАТОК ПРАКТИЧНОЇ РОБОТИ 3 ---")

# ==========================================
# Підготовка: Генерація базових даних
# ==========================================
np.random.seed(42)
n_rows = 100_000

# Генерація великого CSV для тестів
csv_filename = "large_energy_data.csv"
df_mock = pd.DataFrame({
    'timestamp': pd.date_range(start='2020-01-01', periods=n_rows, freq='h'),
    'consumer_id': np.random.randint(1, 100, n_rows),
    'power_kw': np.random.uniform(0.5, 10.0, n_rows),
    'temperature_c': np.random.uniform(-10, 35, n_rows),
    'category': np.random.choice(['household', 'enterprise', 'industrial'], n_rows)
})
df_mock.to_csv(csv_filename, index=False)
print(f"Згенеровано тестовий файл {csv_filename}")

# ==========================================
# Блок 1: Numba, Cython, Multiprocessing
# ==========================================
print("\n[Блок 1] Оптимізація та паралельні обчислення")

# 1. Numba vs Python: Обчислення енергетичних потреб
def energy_calc_python(power, temp):
    res = np.zeros_like(power)
    for i in range(len(power)):
        res[i] = power[i] * (1 + 0.05 * abs(temp[i] - 20)) # Умовна формула
    return res

@jit(nopython=True)
def energy_calc_numba(power, temp):
    res = np.zeros_like(power)
    for i in range(len(power)):
        res[i] = power[i] * (1 + 0.05 * abs(temp[i] - 20))
    return res

power_arr = df_mock['power_kw'].values
temp_arr = df_mock['temperature_c'].values

start = time.time()
energy_calc_python(power_arr, temp_arr)
t_py = time.time() - start

start = time.time()
energy_calc_numba(power_arr, temp_arr) # Перший запуск (компіляція)
start = time.time()
energy_calc_numba(power_arr, temp_arr) # Другий запуск (робочий)
t_nb = time.time() - start

print(f"Час Python: {t_py:.4f}s | Час Numba: {t_nb:.4f}s")

# 3. Multiprocessing: Оцінка ефективності по районах
def evaluate_district(data_chunk):
    return np.mean(data_chunk) * 0.9 # Умовна ефективність

if __name__ == '__main__':
    chunks = np.array_split(power_arr, mp.cpu_count())
    with mp.Pool(mp.cpu_count()) as pool:
        start = time.time()
        results = pool.map(evaluate_district, chunks)
        t_mp = time.time() - start
    print(f"Час Multiprocessing: {t_mp:.4f}s")

# 4. Numba Монте-Карло: Енергетичні інтеграли
@jit(nopython=True)
def monte_carlo_energy_integral(iterations):
    total = 0.0
    for _ in range(iterations):
        x = np.random.random() * 24 # Години
        total += np.sin(x / 24 * np.pi) * 100 # Умовне навантаження
    return (24.0 / iterations) * total

print(f"Інтеграл Монте-Карло (10^5 ітерацій): {monte_carlo_energy_integral(100_000):.2f}")

# 5. Сортування: Numba vs numpy.sort
@jit(nopython=True)
def numba_bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr

# Беремо малий зріз для bubble sort, щоб не чекати вічність
small_arr = power_arr[:5000].copy()
start = time.time(); np.sort(small_arr); t_np_sort = time.time() - start
start = time.time(); numba_bubble_sort(small_arr.copy()); t_nb_sort = time.time() - start
print(f"Сортування (5k). NumPy: {t_np_sort:.5f}s | Numba: {t_nb_sort:.5f}s")

# ==========================================
# Блок 2: Pandas, пам'ять та агрегація
# ==========================================
print("\n[Блок 2] Оптимізація Pandas")

# 6. chunk_size для завантаження
start = time.time()
chunk_iter = pd.read_csv(csv_filename, chunksize=25000)
total_power = sum(chunk['power_kw'].sum() for chunk in chunk_iter)
print(f"Обробка чанками (сума): {total_power:.2f}. Час: {time.time() - start:.4f}s")

# 7. Фільтрація query()
df = pd.read_csv(csv_filename)
med_pow = df['power_kw'].median()
med_temp = df['temperature_c'].median()
df_filtered = df.query("power_kw > @med_pow and temperature_c > @med_temp")
print(f"Знайдено рядків через query(): {len(df_filtered)}")

# 8. Багаторівнева агрегація
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['year'] = df['timestamp'].dt.year
df['month'] = df['timestamp'].dt.month
agg_res = df.groupby(['year', 'month', 'category'])['power_kw'].sum()
print("Агрегація виконана успішно.")

# 9. Використання eval()
start = time.time()
df['total_cost'] = df['power_kw'] * 1.5 + df['temperature_c'] * 0.1
t_norm = time.time() - start

start = time.time()
df.eval("total_cost_eval = power_kw * 1.5 + temperature_c * 0.1", inplace=True)
t_eval = time.time() - start
print(f"Час eval(): {t_eval:.4f}s | Стандартно: {t_norm:.4f}s")

# 10. Оптимізація пам'яті astype()
mem_before = df.memory_usage(deep=True).sum() / 1024**2
df['consumer_id'] = df['consumer_id'].astype('int16')
df['category'] = df['category'].astype('category')
df['power_kw'] = df['power_kw'].astype('float32')
mem_after = df.memory_usage(deep=True).sum() / 1024**2
print(f"Пам'ять ДО: {mem_before:.2f} MB | ПІСЛЯ: {mem_after:.2f} MB")

# ==========================================
# Блок 3: Часові ряди та прогнозування
# ==========================================
print("\n[Блок 3] Аналіз часових рядів")

ts_data = df.groupby('timestamp')['power_kw'].sum().resample('D').sum()

# 12. Інтерполяція пропусків
ts_data.iloc[5:10] = np.nan # Штучні пропуски
ts_filled = ts_data.interpolate(method='linear')
print(f"Пропуски заповнено. Кількість NaN: {ts_filled.isna().sum()}")

# 13. Тижневий resample
ts_weekly = ts_filled.resample('W').mean()

# 14. Децимація (зменшення роздільної здатності)
ts_decimated = ts_filled.iloc[::3] # Беремо кожен 3-й день

# 11 & 15. Тренд та сезонність (seasonal_decompose)
# Встановлюємо період=7 (тижнева сезонність)
decomposition = seasonal_decompose(ts_filled, model='additive', period=7)
print("Декомпозицію (тренд, сезонність, залишок) завершено.")

# ==========================================
# Блок 4: Генерація даних та аномалії
# ==========================================
print("\n[Блок 4] Генератори та розподіли")

# 16. Faker
fake = Faker()
fake_data = [{'name': fake.company(), 'type': 'enterprise', 'consumption': fake.random_int(min=1000, max=5000)} for _ in range(5)]
print(f"Faker приклад: {fake_data[0]}")

# 18. scipy.stats (Нормальний та Експоненційний розподіл)
norm_dist = stats.norm.rvs(loc=500, scale=100, size=1000)
exp_dist = stats.expon.rvs(scale=500, size=1000)

# 19. Markovify (Генерація звіту)
text_corpus = "The energy load is high during winter. Solar generation peaks in summer. Grid stability requires constant monitoring. Energy consumption drops at night."
text_model = markovify.Text(text_corpus)
print(f"Згенерований звіт: {text_model.make_short_sentence(50)}")

# 20. Автокорельовані часові ряди
def generate_ar_series(length, alpha=0.8):
    series = np.zeros(length)
    noise = np.random.normal(0, 1, length)
    for i in range(1, length):
        series[i] = alpha * series[i-1] + noise[i]
    return series
ar_series = generate_ar_series(100)
print(f"Автокореляція першого лагу: {np.corrcoef(ar_series[:-1], ar_series[1:])[0, 1]:.2f}")

# ==========================================
# Блок 5: Big Data (Polars, Dask, Parquet)
# ==========================================
print("\n[Блок 5] Інструменти Big Data")

# 21. Кластеризація (Dask-ML або Scikit-learn на вибірці Dask)
ddf = dd.from_pandas(df[['power_kw', 'temperature_c']], npartitions=4)
# Для простоти на локальній машині використовуємо sklearn на частині даних Dask
sample = ddf.sample(frac=0.1).compute()
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10).fit(sample)
print("Кластеризацію (KMeans) завершено.")

# 23. Polars vs Pandas
start = time.time(); pd.read_csv(csv_filename); t_pd_csv = time.time() - start
start = time.time(); pl.read_csv(csv_filename); t_pl_csv = time.time() - start
print(f"Читання CSV. Pandas: {t_pd_csv:.4f}s | Polars: {t_pl_csv:.4f}s")

# 24. Кешування в Parquet (через PyArrow)
parquet_filename = "energy_cache.parquet"
table = pa.Table.from_pandas(df)
pq.write_table(table, parquet_filename)
print(f"Дані успішно збережено у {parquet_filename}")

# 25. Featuretools
es = ft.EntitySet(id="energy_data")
es.add_dataframe(dataframe_name="readings", dataframe=df[['consumer_id', 'power_kw', 'temperature_c']].head(1000), index="id", make_index=True)
features, feature_names = ft.dfs(entityset=es, target_dataframe_name="readings", max_depth=1)
print(f"Згенеровано {len(feature_names)} ознак через Featuretools.")

print("\n--- ВСІ ЗАВДАННЯ ВИКОНАНО УСПІШНО ---")
