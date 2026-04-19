import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

def generate_sample_data(filename):
    """Генерує тестовий датасет, якщо реального файлу ще немає."""
    np.random.seed(42)
    locations = ['Bedroom', 'Living Room', 'TV', 'Microwave', 'Substation']
    data = []
    
    for loc in locations:
        if loc == 'Bedroom' or loc == 'Living Room':
            base_mag = np.random.normal(45, 5, 200) 
        elif loc == 'TV':
            base_mag = np.random.normal(80, 15, 200)
        elif loc == 'Microwave':
            base_mag = np.random.normal(150, 30, 200)
            base_mag[np.random.randint(0, 200, 5)] += 200 
        else: 
            base_mag = np.random.normal(120, 20, 200)
            
        for i in range(200):
            data.append({
                'Time': len(data) * 0.1,
                'Magnetic_Field_X': base_mag[i] * 0.5 + np.random.normal(0, 2),
                'Magnetic_Field_Y': base_mag[i] * 0.5 + np.random.normal(0, 2),
                'Magnetic_Field_Z': base_mag[i] * 0.7 + np.random.normal(0, 2),
                'Location': loc
            })
            
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"[INFO] Створено тестовий файл: {filename}")

def load_and_preprocess(filename):
    """Завантажує дані та розраховує абсолютне значення магнітного поля."""
    if not os.path.exists(filename):
        generate_sample_data(filename)
        
    df = pd.read_csv(filename)
    
    if 'Absolute_Magnitude' not in df.columns:
        df['Absolute_Magnitude'] = np.sqrt(
            df['Magnetic_Field_X']**2 + 
            df['Magnetic_Field_Y']**2 + 
            df['Magnetic_Field_Z']**2
        )
    return df

def detect_anomalies(df):
    """Виявляє аномалії за допомогою Z-оцінки (Z-score > 3)."""
    df['Z_Score'] = np.abs(stats.zscore(df['Absolute_Magnitude']))
    anomalies = df[df['Z_Score'] > 3]
    print(f"\n[АНОМАЛІЇ] Знайдено {len(anomalies)} аномальних сплесків магнітного поля.")
    if not anomalies.empty:
        print(anomalies[['Time', 'Location', 'Absolute_Magnitude']].head())
    return df, anomalies

def plot_comparative_boxplot(df):
    """Будує боксплот для порівняння рівня поля в різних зонах."""
    plt.figure()
    sns.boxplot(x='Location', y='Absolute_Magnitude', data=df, palette='viridis')
    plt.title('Порівняння рівня магнітного поля у різних зонах')
    plt.ylabel('Магнітна індукція (мкТл)')
    plt.xlabel('Локація')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('magnetic_comparison_boxplot.png')
    plt.show()

def plot_time_series_with_anomalies(df, anomalies):
    """Будує часовий ряд із позначенням аномальних сплесків."""
    plt.figure()
    sns.lineplot(x='Time', y='Absolute_Magnitude', hue='Location', data=df, alpha=0.7)
    
    if not anomalies.empty:
        plt.scatter(anomalies['Time'], anomalies['Absolute_Magnitude'], color='red', s=50, label='Аномалії', zorder=5)
        
    plt.title('Динаміка магнітного поля у часі')
    plt.ylabel('Магнітна індукція (мкТл)')
    plt.xlabel('Час (с)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('magnetic_time_series.png')
    plt.show()

def create_magnetic_heatmap(df):
    """Створює спрощену карту (теплову карту) магнітного поля будинку."""
    mean_values = df.groupby('Location')['Absolute_Magnitude'].mean().reset_index()
    
    layout = {
        'Bedroom': (0, 0),
        'Living Room': (0, 1),
        'TV': (0, 2),
        'Microwave': (1, 1),
        'Substation': (2, 2)
    }
    
    grid = np.zeros((3, 3))
    grid[:] = np.nan 
    
    for _, row in mean_values.iterrows():
        loc = row['Location']
        if loc in layout:
            x, y = layout[loc]
            grid[x, y] = row['Absolute_Magnitude']
            
    plt.figure(figsize=(8, 6))
    sns.heatmap(grid, annot=True, fmt=".1f", cmap="YlOrRd", cbar_kws={'label': 'мкТл'}, 
                xticklabels=['Зона 1', 'Зона 2', 'Зона 3'], 
                yticklabels=['Північ', 'Центр', 'Південь'])
    
    plt.title('Теплова карта магнітного поля (План-схема)')
    plt.tight_layout()
    plt.savefig('magnetic_heatmap.png')
    plt.show()

if __name__ == "__main__":
    file_path = 'magnetic_data.csv'
    
    df = load_and_preprocess(file_path)
    
    df, anomalies = detect_anomalies(df)
    
    plot_comparative_boxplot(df)
    plot_time_series_with_anomalies(df, anomalies)
    create_magnetic_heatmap(df)
    
    print("\n[УСПІХ] Аналіз завершено. Графіки збережено у поточну папку.")
