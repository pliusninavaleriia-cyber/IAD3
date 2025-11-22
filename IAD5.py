"""
Лабораторна робота: AdaBoostRegressor для прогнозування цін акцій Google
Датасет: https://www.kaggle.com/datasets/rahulsah06/gooogle-stock-price

Інструкція:
1. Завантажте Google_Stock_Price_Train.csv з Kaggle
2. Розмістіть файл в тій же папці, що й цей скрипт
3. Запустіть скрипт
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
import time
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# 1. ЗАВАНТАЖЕННЯ ТА ВІЗУАЛІЗАЦІЯ ДАНИХ
# =============================================================================
print("=" * 70)
print("1. ЗАВАНТАЖЕННЯ ТА ВІЗУАЛІЗАЦІЯ ПОЧАТКОВИХ 2D ДАНИХ")
print("=" * 70)

# Спроба завантажити дані з файлу
try:
    data = pd.read_csv('Google_Stock_Price_Train.csv')
    print("Дані успішно завантажено з файлу!")
    
    # Очищення даних - видалення ком у числах
    for col in ['Open', 'High', 'Low', 'Close']:
        if data[col].dtype == 'object':
            data[col] = pd.to_numeric(data[col].str.replace(',', ''), errors='coerce')
    if 'Volume' in data.columns and data['Volume'].dtype == 'object':
        data['Volume'] = pd.to_numeric(data['Volume'].str.replace(',', ''), errors='coerce')
    
    data = data.dropna()
    
except FileNotFoundError:
    print("Файл не знайдено. Генеруємо синтетичні дані для демонстрації...")
    np.random.seed(42)
    n_samples = 1000
    
    # Симуляція цін акцій Google
    base_price = 325
    trend = np.linspace(0, 400, n_samples)
    noise = np.cumsum(np.random.randn(n_samples) * 3)
    seasonality = 25 * np.sin(np.linspace(0, 8*np.pi, n_samples))
    close_prices = base_price + trend + noise + seasonality
    close_prices = np.maximum(close_prices, 100)
    
    data = pd.DataFrame({
        'Date': pd.date_range(start='2012-01-03', periods=n_samples, freq='B'),
        'Open': close_prices * (1 + np.random.randn(n_samples)*0.01),
        'High': close_prices * (1 + np.abs(np.random.randn(n_samples)*0.02)),
        'Low': close_prices * (1 - np.abs(np.random.randn(n_samples)*0.02)),
        'Close': close_prices,
        'Volume': np.random.randint(1000000, 10000000, n_samples)
    })

print(f"\nРозмір даних: {data.shape}")
print(f"\nПерші 5 рядків:")
print(data.head())
print(f"\nСтатистика:")
print(data.describe())

# Створення індексу дня для 2D візуалізації
data['Day_Index'] = np.arange(len(data))

# Візуалізація початкових даних
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Графік 1: Ціна закриття від індексу дня
ax1 = axes[0, 0]
ax1.scatter(data['Day_Index'], data['Close'], alpha=0.6, s=15, c='blue')
ax1.set_xlabel('День (індекс)')
ax1.set_ylabel('Ціна закриття ($)')
ax1.set_title('Ціна закриття акцій Google')
ax1.grid(True, alpha=0.3)

# Графік 2: Open vs Close
ax2 = axes[0, 1]
ax2.scatter(data['Open'], data['Close'], alpha=0.5, s=15, c='green')
ax2.set_xlabel('Ціна відкриття ($)')
ax2.set_ylabel('Ціна закриття ($)')
ax2.set_title('Залежність Close від Open')
ax2.grid(True, alpha=0.3)

# Графік 3: High vs Low
ax3 = axes[1, 0]
ax3.scatter(data['High'], data['Low'], alpha=0.5, s=15, c='purple')
ax3.set_xlabel('Максимальна ціна ($)')
ax3.set_ylabel('Мінімальна ціна ($)')
ax3.set_title('Залежність Low від High')
ax3.grid(True, alpha=0.3)

# Графік 4: Розподіл ціни закриття
ax4 = axes[1, 1]
ax4.hist(data['Close'], bins=30, alpha=0.7, color='teal', edgecolor='black')
ax4.set_xlabel('Ціна закриття ($)')
ax4.set_ylabel('Частота')
ax4.set_title('Розподіл ціни закриття')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('1_initial_data_visualization.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nГрафік збережено: 1_initial_data_visualization.png")

# =============================================================================
# 2. РОЗБИТТЯ ДАНИХ НА НАВЧАЛЬНИЙ, ПЕРЕВІРОЧНИЙ ТА ТЕСТОВИЙ НАБОРИ
# =============================================================================
print("\n" + "=" * 70)
print("2. РОЗБИТТЯ ДАНИХ")
print("=" * 70)

# Підготовка ознак (2D: Day_Index та Open)
X = data[['Day_Index', 'Open']].values
y = data['Close'].values

# Розбиття: 60% train, 20% validation, 20% test
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)

print(f"Розмір навчального набору: {X_train.shape[0]} зразків")
print(f"Розмір перевірочного набору: {X_val.shape[0]} зразків")
print(f"Розмір тестового набору: {X_test.shape[0]} зразків")
print(f"Загалом: {X_train.shape[0] + X_val.shape[0] + X_test.shape[0]} зразків")

# Масштабування даних
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_val_scaled = scaler_X.transform(X_val)
X_test_scaled = scaler_X.transform(X_test)

y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
y_val_scaled = scaler_y.transform(y_val.reshape(-1, 1)).ravel()
y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).ravel()

print("\nДані масштабовано за допомогою StandardScaler")

# =============================================================================
# 3. ДОСЛІДЖЕННЯ ГІПЕРПАРАМЕТРІВ AdaBoostRegressor
# =============================================================================
print("\n" + "=" * 70)
print("3. ДОСЛІДЖЕННЯ ГІПЕРПАРАМЕТРІВ AdaBoostRegressor")
print("=" * 70)

# Функція для обчислення метрик
def calc_metrics(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    return {'R2': r2, 'RMSE': rmse, 'MAPE': mape}

# 3.1 Базова модель (DecisionTreeRegressor)
print("\n3.1 Базова модель (DecisionTreeRegressor з глибиною 4)")
print("-" * 50)

base_model = DecisionTreeRegressor(max_depth=4, random_state=42)
base_model.fit(X_train_scaled, y_train_scaled)

y_pred_base_val = base_model.predict(X_val_scaled)
y_pred_base_test = base_model.predict(X_test_scaled)

base_val_metrics = calc_metrics(y_val_scaled, y_pred_base_val)
base_test_metrics = calc_metrics(y_test_scaled, y_pred_base_test)

print(f"Validation: R²={base_val_metrics['R2']:.4f}, RMSE={base_val_metrics['RMSE']:.4f}, MAPE={base_val_metrics['MAPE']:.2f}%")
print(f"Test:       R²={base_test_metrics['R2']:.4f}, RMSE={base_test_metrics['RMSE']:.4f}, MAPE={base_test_metrics['MAPE']:.2f}%")

# -----------------------------------------------------------------------------
# 3.2 Дослідження n_estimators
# -----------------------------------------------------------------------------
print("\n3.2 Дослідження параметра n_estimators")
print("-" * 50)

n_estimators_values = [1, 5, 10, 25, 50, 100, 150, 200, 300]
n_est_results = []

for n_est in n_estimators_values:
    ada = AdaBoostRegressor(
        estimator=DecisionTreeRegressor(max_depth=4, random_state=42),
        n_estimators=n_est,
        learning_rate=1.0,
        loss='linear',
        random_state=42
    )
    ada.fit(X_train_scaled, y_train_scaled)
    y_pred = ada.predict(X_val_scaled)
    metrics = calc_metrics(y_val_scaled, y_pred)
    n_est_results.append({'n_estimators': n_est, **metrics})
    print(f"n_estimators={n_est:3d}: R²={metrics['R2']:.4f}, RMSE={metrics['RMSE']:.4f}")

# Графік залежності від n_estimators
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
n_est_df = pd.DataFrame(n_est_results)

axes[0].plot(n_est_df['n_estimators'], n_est_df['R2'], 'b-o', linewidth=2, markersize=6, label='AdaBoost')
axes[0].axhline(y=base_val_metrics['R2'], color='r', linestyle='--', linewidth=2, label='Базова модель')
axes[0].set_xlabel('n_estimators')
axes[0].set_ylabel('R²')
axes[0].set_title('R² vs n_estimators')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(n_est_df['n_estimators'], n_est_df['RMSE'], 'g-o', linewidth=2, markersize=6, label='AdaBoost')
axes[1].axhline(y=base_val_metrics['RMSE'], color='r', linestyle='--', linewidth=2, label='Базова модель')
axes[1].set_xlabel('n_estimators')
axes[1].set_ylabel('RMSE')
axes[1].set_title('RMSE vs n_estimators')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

axes[2].plot(n_est_df['n_estimators'], n_est_df['MAPE'], 'm-o', linewidth=2, markersize=6, label='AdaBoost')
axes[2].axhline(y=base_val_metrics['MAPE'], color='r', linestyle='--', linewidth=2, label='Базова модель')
axes[2].set_xlabel('n_estimators')
axes[2].set_ylabel('MAPE (%)')
axes[2].set_title('MAPE vs n_estimators')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('2_n_estimators_analysis.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nГрафік збережено: 2_n_estimators_analysis.png")

# -----------------------------------------------------------------------------
# 3.3 Дослідження learning_rate
# -----------------------------------------------------------------------------
print("\n3.3 Дослідження параметра learning_rate")
print("-" * 50)

learning_rates = [0.001, 0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
lr_results = []

for lr in learning_rates:
    ada = AdaBoostRegressor(
        estimator=DecisionTreeRegressor(max_depth=4, random_state=42),
        n_estimators=50,
        learning_rate=lr,
        loss='linear',
        random_state=42
    )
    ada.fit(X_train_scaled, y_train_scaled)
    y_pred = ada.predict(X_val_scaled)
    metrics = calc_metrics(y_val_scaled, y_pred)
    lr_results.append({'learning_rate': lr, **metrics})
    print(f"learning_rate={lr:.3f}: R²={metrics['R2']:.4f}, RMSE={metrics['RMSE']:.4f}")

# Графік
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
lr_df = pd.DataFrame(lr_results)

axes[0].semilogx(lr_df['learning_rate'], lr_df['R2'], 'b-o', linewidth=2, markersize=6, label='AdaBoost')
axes[0].axhline(y=base_val_metrics['R2'], color='r', linestyle='--', linewidth=2, label='Базова модель')
axes[0].set_xlabel('learning_rate')
axes[0].set_ylabel('R²')
axes[0].set_title('R² vs learning_rate')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].semilogx(lr_df['learning_rate'], lr_df['RMSE'], 'g-o', linewidth=2, markersize=6, label='AdaBoost')
axes[1].axhline(y=base_val_metrics['RMSE'], color='r', linestyle='--', linewidth=2, label='Базова модель')
axes[1].set_xlabel('learning_rate')
axes[1].set_ylabel('RMSE')
axes[1].set_title('RMSE vs learning_rate')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

axes[2].semilogx(lr_df['learning_rate'], lr_df['MAPE'], 'm-o', linewidth=2, markersize=6, label='AdaBoost')
axes[2].axhline(y=base_val_metrics['MAPE'], color='r', linestyle='--', linewidth=2, label='Базова модель')
axes[2].set_xlabel('learning_rate')
axes[2].set_ylabel('MAPE (%)')
axes[2].set_title('MAPE vs learning_rate')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('3_learning_rate_analysis.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nГрафік збережено: 3_learning_rate_analysis.png")

# -----------------------------------------------------------------------------
# 3.4 Дослідження loss функції
# -----------------------------------------------------------------------------
print("\n3.4 Дослідження параметра loss")
print("-" * 50)

loss_functions = ['linear', 'square', 'exponential']
loss_results = []

for loss in loss_functions:
    ada = AdaBoostRegressor(
        estimator=DecisionTreeRegressor(max_depth=4, random_state=42),
        n_estimators=50,
        learning_rate=0.5,
        loss=loss,
        random_state=42
    )
    ada.fit(X_train_scaled, y_train_scaled)
    y_pred = ada.predict(X_val_scaled)
    metrics = calc_metrics(y_val_scaled, y_pred)
    loss_results.append({'loss': loss, **metrics})
    print(f"loss='{loss:12s}': R²={metrics['R2']:.4f}, RMSE={metrics['RMSE']:.4f}, MAPE={metrics['MAPE']:.2f}%")

# Графік
fig, ax = plt.subplots(figsize=(10, 6))
loss_df = pd.DataFrame(loss_results)
x = np.arange(len(loss_functions))
width = 0.25

bars1 = ax.bar(x - width, loss_df['R2'], width, label='R²', color='steelblue')
bars2 = ax.bar(x, 1 - loss_df['RMSE']/loss_df['RMSE'].max(), width, label='1 - norm(RMSE)', color='seagreen')
bars3 = ax.bar(x + width, 1 - loss_df['MAPE']/loss_df['MAPE'].max(), width, label='1 - norm(MAPE)', color='coral')

ax.set_xlabel('Loss функція')
ax.set_ylabel('Нормалізований показник (більше = краще)')
ax.set_title('Порівняння loss функцій AdaBoostRegressor')
ax.set_xticks(x)
ax.set_xticklabels(loss_functions)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('4_loss_function_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nГрафік збережено: 4_loss_function_comparison.png")

# -----------------------------------------------------------------------------
# 3.5 Комбінований пошук гіперпараметрів
# -----------------------------------------------------------------------------
print("\n3.5 Комбінований пошук оптимальних гіперпараметрів")
print("-" * 50)

best_r2 = -np.inf
best_params = {}

n_est_grid = [25, 50, 100, 150]
lr_grid = [0.1, 0.3, 0.5, 1.0]
loss_grid = ['linear', 'square', 'exponential']

print("Пошук найкращої комбінації параметрів...")
for n_est in n_est_grid:
    for lr in lr_grid:
        for loss in loss_grid:
            ada = AdaBoostRegressor(
                estimator=DecisionTreeRegressor(max_depth=4, random_state=42),
                n_estimators=n_est,
                learning_rate=lr,
                loss=loss,
                random_state=42
            )
            ada.fit(X_train_scaled, y_train_scaled)
            y_pred = ada.predict(X_val_scaled)
            r2 = r2_score(y_val_scaled, y_pred)
            
            if r2 > best_r2:
                best_r2 = r2
                best_params = {'n_estimators': n_est, 'learning_rate': lr, 'loss': loss}

print(f"\nНайкращі параметри:")
print(f"  n_estimators: {best_params['n_estimators']}")
print(f"  learning_rate: {best_params['learning_rate']}")
print(f"  loss: {best_params['loss']}")
print(f"  R² на валідації: {best_r2:.4f}")

# -----------------------------------------------------------------------------
# 3.6 Порівняння різних базових моделей
# -----------------------------------------------------------------------------
print("\n3.6 Порівняння різних базових моделей (base_estimator)")
print("-" * 50)

base_estimators = {
    'DecisionTree(depth=3)': DecisionTreeRegressor(max_depth=3, random_state=42),
    'DecisionTree(depth=4)': DecisionTreeRegressor(max_depth=4, random_state=42),
    'DecisionTree(depth=5)': DecisionTreeRegressor(max_depth=5, random_state=42),
    'DecisionTree(depth=6)': DecisionTreeRegressor(max_depth=6, random_state=42),
    'LinearRegression': LinearRegression(),
}

estimator_results = []
for name, estimator in base_estimators.items():
    # Індивідуальна модель
    est_copy = type(estimator)(**estimator.get_params())
    est_copy.fit(X_train_scaled, y_train_scaled)
    y_pred_ind = est_copy.predict(X_val_scaled)
    ind_r2 = r2_score(y_val_scaled, y_pred_ind)
    
    # AdaBoost з цією базовою моделлю
    ada = AdaBoostRegressor(
        estimator=estimator,
        n_estimators=50,
        learning_rate=0.5,
        loss='linear',
        random_state=42
    )
    ada.fit(X_train_scaled, y_train_scaled)
    y_pred_ada = ada.predict(X_val_scaled)
    ada_r2 = r2_score(y_val_scaled, y_pred_ada)
    
    estimator_results.append({
        'Estimator': name,
        'Individual_R2': ind_r2,
        'AdaBoost_R2': ada_r2,
        'Improvement': ada_r2 - ind_r2
    })
    print(f"{name:25s}: Індивід. R²={ind_r2:.4f}, AdaBoost R²={ada_r2:.4f}, Δ={ada_r2-ind_r2:+.4f}")

# Графік
fig, ax = plt.subplots(figsize=(12, 6))
est_df = pd.DataFrame(estimator_results)
x = np.arange(len(est_df))
width = 0.35

bars1 = ax.bar(x - width/2, est_df['Individual_R2'], width, label='Індивідуальна модель', color='lightcoral')
bars2 = ax.bar(x + width/2, est_df['AdaBoost_R2'], width, label='AdaBoost ансамбль', color='steelblue')

ax.set_xlabel('Базова модель')
ax.set_ylabel('R²')
ax.set_title('Порівняння індивідуальних моделей та AdaBoost ансамблів')
ax.set_xticks(x)
ax.set_xticklabels(est_df['Estimator'], rotation=15, ha='right')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

for bar in bars1:
    h = bar.get_height()
    ax.annotate(f'{h:.3f}', xy=(bar.get_x() + bar.get_width()/2, h),
                xytext=(0, 3), textcoords="offset points", ha='center', fontsize=8)
for bar in bars2:
    h = bar.get_height()
    ax.annotate(f'{h:.3f}', xy=(bar.get_x() + bar.get_width()/2, h),
                xytext=(0, 3), textcoords="offset points", ha='center', fontsize=8)

plt.tight_layout()
plt.savefig('5_base_estimator_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nГрафік збережено: 5_base_estimator_comparison.png")

# =============================================================================
# 4. ВІЗУАЛІЗАЦІЯ ПРОГНОЗІВ
# =============================================================================
print("\n" + "=" * 70)
print("4. ВІЗУАЛІЗАЦІЯ ПРОГНОЗІВ")
print("=" * 70)

# Навчання найкращої моделі
best_ada = AdaBoostRegressor(
    estimator=DecisionTreeRegressor(max_depth=4, random_state=42),
    n_estimators=best_params['n_estimators'],
    learning_rate=best_params['learning_rate'],
    loss=best_params['loss'],
    random_state=42
)
best_ada.fit(X_train_scaled, y_train_scaled)

# Прогнози (масштабовані)
y_pred_ada_test_scaled = best_ada.predict(X_test_scaled)
y_pred_base_test_scaled = base_model.predict(X_test_scaled)

# Зворотне перетворення масштабу для візуалізації
y_test_orig = scaler_y.inverse_transform(y_test_scaled.reshape(-1, 1)).ravel()
y_pred_ada_orig = scaler_y.inverse_transform(y_pred_ada_test_scaled.reshape(-1, 1)).ravel()
y_pred_base_orig = scaler_y.inverse_transform(y_pred_base_test_scaled.reshape(-1, 1)).ravel()

# Сортування за індексом дня
sort_idx = np.argsort(X_test[:, 0])

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Графік 1: Прогнози vs Реальні значення
ax1 = axes[0, 0]
ax1.scatter(X_test[sort_idx, 0], y_test_orig[sort_idx], alpha=0.6, s=30, 
            c='black', label='Реальні значення', zorder=3)
ax1.plot(X_test[sort_idx, 0], y_pred_ada_orig[sort_idx], 'b-', 
         linewidth=2, label='AdaBoost прогноз', zorder=2)
ax1.plot(X_test[sort_idx, 0], y_pred_base_orig[sort_idx], 'r--', 
         linewidth=2, label='Базова модель', zorder=1)
ax1.set_xlabel('День (індекс)')
ax1.set_ylabel('Ціна закриття ($)')
ax1.set_title('Прогнози на тестовому наборі')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Графік 2: Scatter plot прогнозів
ax2 = axes[0, 1]
ax2.scatter(y_test_orig, y_pred_ada_orig, alpha=0.6, s=30, c='blue', label='AdaBoost')
ax2.scatter(y_test_orig, y_pred_base_orig, alpha=0.4, s=30, c='red', label='Базова модель')
min_v = min(y_test_orig.min(), y_pred_ada_orig.min())
max_v = max(y_test_orig.max(), y_pred_ada_orig.max())
ax2.plot([min_v, max_v], [min_v, max_v], 'k--', linewidth=2, label='Ідеальний прогноз')
ax2.set_xlabel('Реальні значення ($)')
ax2.set_ylabel('Прогнозовані значення ($)')
ax2.set_title('Scatter plot: Прогноз vs Реальність')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Графік 3: Помилки прогнозування
ax3 = axes[1, 0]
errors_ada = y_test_orig - y_pred_ada_orig
errors_base = y_test_orig - y_pred_base_orig
ax3.hist(errors_ada, bins=25, alpha=0.6, label=f'AdaBoost (σ={np.std(errors_ada):.2f})', color='blue')
ax3.hist(errors_base, bins=25, alpha=0.6, label=f'Базова (σ={np.std(errors_base):.2f})', color='red')
ax3.axvline(x=0, color='black', linestyle='--', linewidth=2)
ax3.set_xlabel('Помилка прогнозування ($)')
ax3.set_ylabel('Частота')
ax3.set_title('Розподіл помилок прогнозування')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Графік 4: Залишки
ax4 = axes[1, 1]
ax4.scatter(y_pred_ada_orig, errors_ada, alpha=0.6, s=30, c='blue', label='AdaBoost')
ax4.scatter(y_pred_base_orig, errors_base, alpha=0.4, s=30, c='red', label='Базова модель')
ax4.axhline(y=0, color='black', linestyle='--', linewidth=2)
ax4.set_xlabel('Прогнозовані значення ($)')
ax4.set_ylabel('Залишки ($)')
ax4.set_title('Графік залишків')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('6_predictions_visualization.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nГрафік збережено: 6_predictions_visualization.png")

# =============================================================================
# 5. РОЗРАХУНОК ЗМІЩЕННЯ ТА ДИСПЕРСІЇ
# =============================================================================
print("\n" + "=" * 70)
print("5. РОЗРАХУНОК ЗМІЩЕННЯ ТА ДИСПЕРСІЇ (Bias-Variance)")
print("=" * 70)

def bias_variance_decomp(model_fn, X_tr, y_tr, X_te, y_te, n_bootstrap=50):
    """Обчислює bias² та variance методом bootstrap"""
    np.random.seed(42)
    n_test = len(X_te)
    preds = np.zeros((n_bootstrap, n_test))
    
    for i in range(n_bootstrap):
        idx = np.random.choice(len(X_tr), size=len(X_tr), replace=True)
        X_b, y_b = X_tr[idx], y_tr[idx]
        
        model = model_fn()
        model.fit(X_b, y_b)
        preds[i] = model.predict(X_te)
    
    mean_pred = np.mean(preds, axis=0)
    bias_sq = np.mean((y_te - mean_pred) ** 2)
    variance = np.mean(np.var(preds, axis=0))
    total_err = bias_sq + variance
    
    return {'bias_squared': bias_sq, 'variance': variance, 'total_error': total_err}

print("\nОбчислення Bias-Variance (це може зайняти деякий час)...")

# Базова модель
print("  - Базова модель (DecisionTree)...")
base_bv = bias_variance_decomp(
    lambda: DecisionTreeRegressor(max_depth=4, random_state=None),
    X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled,
    n_bootstrap=30
)

# AdaBoost
print("  - AdaBoost ансамбль...")
def create_adaboost():
    return AdaBoostRegressor(
        estimator=DecisionTreeRegressor(max_depth=4),
        n_estimators=best_params['n_estimators'],
        learning_rate=best_params['learning_rate'],
        loss=best_params['loss'],
        random_state=None
    )

ada_bv = bias_variance_decomp(
    create_adaboost,
    X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled,
    n_bootstrap=30
)

print("\n" + "-" * 60)
print("РЕЗУЛЬТАТИ BIAS-VARIANCE ДЕКОМПОЗИЦІЇ")
print("-" * 60)
print(f"{'Модель':<25} {'Bias²':<12} {'Variance':<12} {'Total Error':<12}")
print("-" * 60)
print(f"{'DecisionTree (базова)':<25} {base_bv['bias_squared']:<12.4f} {base_bv['variance']:<12.4f} {base_bv['total_error']:<12.4f}")
print(f"{'AdaBoostRegressor':<25} {ada_bv['bias_squared']:<12.4f} {ada_bv['variance']:<12.4f} {ada_bv['total_error']:<12.4f}")
print("-" * 60)

# Візуалізація
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

models_names = ['DecisionTree\n(базова)', 'AdaBoost\n(ансамбль)']
bias_vals = [base_bv['bias_squared'], ada_bv['bias_squared']]
var_vals = [base_bv['variance'], ada_bv['variance']]
total_vals = [base_bv['total_error'], ada_bv['total_error']]

# Графік 1: Порівняння компонентів
ax1 = axes[0]
x = np.arange(len(models_names))
width = 0.35

b1 = ax1.bar(x - width/2, bias_vals, width, label='Bias²', color='coral')
b2 = ax1.bar(x + width/2, var_vals, width, label='Variance', color='steelblue')

ax1.set_ylabel('Значення')
ax1.set_title('Bias-Variance декомпозиція')
ax1.set_xticks(x)
ax1.set_xticklabels(models_names)
ax1.legend()
ax1.grid(True, alpha=0.3, axis='y')

for bar in b1:
    h = bar.get_height()
    ax1.annotate(f'{h:.4f}', xy=(bar.get_x() + bar.get_width()/2, h),
                xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9)
for bar in b2:
    h = bar.get_height()
    ax1.annotate(f'{h:.4f}', xy=(bar.get_x() + bar.get_width()/2, h),
                xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9)

# Графік 2: Стековий графік
ax2 = axes[1]
ax2.bar(models_names, bias_vals, label='Bias²', color='coral')
ax2.bar(models_names, var_vals, bottom=bias_vals, label='Variance', color='steelblue')
ax2.set_ylabel('Total Error')
ax2.set_title('Загальна помилка = Bias² + Variance')
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')

for i, t in enumerate(total_vals):
    ax2.annotate(f'Total: {t:.4f}', xy=(i, t), xytext=(0, 5),
                textcoords="offset points", ha='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('7_bias_variance_analysis.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nГрафік збережено: 7_bias_variance_analysis.png")

# =============================================================================
# 6. АНАЛІЗ ЧАСУ НАВЧАННЯ
# =============================================================================
print("\n" + "=" * 70)
print("6. АНАЛІЗ ЧАСУ НАВЧАННЯ")
print("=" * 70)

timing_results = []

# Базова модель
t_start = time.time()
for _ in range(20):
    m = DecisionTreeRegressor(max_depth=4, random_state=42)
    m.fit(X_train_scaled, y_train_scaled)
base_time = (time.time() - t_start) / 20
timing_results.append({'Model': 'DecisionTree (базова)', 'Time_sec': base_time, 'n_est': 1})
print(f"DecisionTree (базова): {base_time*1000:.3f} мс")

# AdaBoost з різною кількістю estimators
for n_est in [10, 25, 50, 100, 150, 200]:
    t_start = time.time()
    for _ in range(5):
        ada = AdaBoostRegressor(
            estimator=DecisionTreeRegressor(max_depth=4),
            n_estimators=n_est,
            learning_rate=0.5,
            random_state=42
        )
        ada.fit(X_train_scaled, y_train_scaled)
    ada_time = (time.time() - t_start) / 5
    timing_results.append({'Model': f'AdaBoost (n={n_est})', 'Time_sec': ada_time, 'n_est': n_est})
    ratio = ada_time / base_time
    print(f"AdaBoost (n_estimators={n_est:3d}): {ada_time*1000:.3f} мс (x{ratio:.1f} від базової)")

# Графік
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
timing_df = pd.DataFrame(timing_results)

# Графік 1: Час навчання барплот
ax1 = axes[0]
colors = ['green'] + ['steelblue'] * (len(timing_df) - 1)
bars = ax1.barh(timing_df['Model'], timing_df['Time_sec'] * 1000, color=colors)
ax1.set_xlabel('Час навчання (мілісекунди)')
ax1.set_title('Порівняння часу навчання моделей')
ax1.grid(True, alpha=0.3, axis='x')

for bar, t in zip(bars, timing_df['Time_sec']):
    ax1.annotate(f'{t*1000:.2f} мс', xy=(t*1000, bar.get_y() + bar.get_height()/2),
                xytext=(5, 0), textcoords="offset points", ha='left', va='center', fontsize=9)

# Графік 2: Залежність часу від n_estimators
ax2 = axes[1]
ada_timing = timing_df[timing_df['n_est'] > 1]
ax2.plot(ada_timing['n_est'], ada_timing['Time_sec'] * 1000, 'b-o', linewidth=2, markersize=8)
ax2.axhline(y=base_time * 1000, color='r', linestyle='--', linewidth=2, label='Базова модель')
ax2.set_xlabel('n_estimators')
ax2.set_ylabel('Час навчання (мілісекунди)')
ax2.set_title('Залежність часу навчання від n_estimators')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('8_training_time_analysis.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nГрафік збережено: 8_training_time_analysis.png")

# =============================================================================
# 7. ФІНАЛЬНА ОЦІНКА ТА ВИСНОВКИ
# =============================================================================
print("\n" + "=" * 70)
print("7. ФІНАЛЬНА ОЦІНКА НА ТЕСТОВОМУ НАБОРІ ТА ВИСНОВКИ")
print("=" * 70)

# Фінальні метрики
ada_final_metrics = calc_metrics(y_test_scaled, y_pred_ada_test_scaled)
base_final_metrics = calc_metrics(y_test_scaled, y_pred_base_test_scaled)

print("\n" + "-" * 65)
print("ФІНАЛЬНІ РЕЗУЛЬТАТИ НА ТЕСТОВОМУ НАБОРІ")
print("-" * 65)
print(f"{'Метрика':<12} {'Базова модель':<18} {'AdaBoost':<18} {'Покращення':<15}")
print("-" * 65)
print(f"{'R²':<12} {base_final_metrics['R2']:<18.4f} {ada_final_metrics['R2']:<18.4f} {ada_final_metrics['R2']-base_final_metrics['R2']:+.4f}")
print(f"{'RMSE':<12} {base_final_metrics['RMSE']:<18.4f} {ada_final_metrics['RMSE']:<18.4f} {ada_final_metrics['RMSE']-base_final_metrics['RMSE']:+.4f}")
print(f"{'MAPE (%)':<12} {base_final_metrics['MAPE']:<18.2f} {ada_final_metrics['MAPE']:<18.2f} {ada_final_metrics['MAPE']-base_final_metrics['MAPE']:+.2f}")
print("-" * 65)

print(f"\nНайкращі гіперпараметри AdaBoost:")
print(f"  • n_estimators: {best_params['n_estimators']}")
print(f"  • learning_rate: {best_params['learning_rate']}")
print(f"  • loss: {best_params['loss']}")

# =============================================================================
# ВИСНОВКИ
# =============================================================================
print("\n" + "=" * 70)
print("ВИСНОВКИ")
print("=" * 70)

r2_improvement = ((ada_final_metrics['R2'] - base_final_metrics['R2']) / 
                  abs(base_final_metrics['R2']) * 100) if base_final_metrics['R2'] != 0 else 0

conclusions = f"""
1. ВПЛИВ ГІПЕРПАРАМЕТРІВ AdaBoostRegressor:

   • n_estimators (кількість базових моделей):
     - Збільшення покращує якість до певної межі (50-100 моделей)
     - Після цього приріст стає мінімальним, а час зростає лінійно
     - Оптимальне значення: {best_params['n_estimators']}
     
   • learning_rate (швидкість навчання):
     - Контролює внесок кожної моделі в ансамбль
     - Занадто високі значення → перенавчання
     - Занадто низькі значення → недонавчання
     - Оптимальне значення: {best_params['learning_rate']}
     
   • loss (функція втрат):
     - 'linear': лінійна втрата, стабільні результати
     - 'square': квадратична, чутлива до викидів
     - 'exponential': експоненціальна, найбільш чутлива
     - Оптимальне значення: {best_params['loss']}

2. ПОРІВНЯННЯ З БАЗОВОЮ МОДЕЛЛЮ:
   
   • AdaBoost R²: {ada_final_metrics['R2']:.4f} vs Базова R²: {base_final_metrics['R2']:.4f}
   • Відносне покращення R²: {r2_improvement:+.1f}%
   • RMSE знизився з {base_final_metrics['RMSE']:.4f} до {ada_final_metrics['RMSE']:.4f}

3. BIAS-VARIANCE АНАЛІЗ:
   
   • Базова модель: Bias²={base_bv['bias_squared']:.4f}, Var={base_bv['variance']:.4f}
   • AdaBoost:      Bias²={ada_bv['bias_squared']:.4f}, Var={ada_bv['variance']:.4f}
   • AdaBoost зменшує зміщення (bias) через послідовне навчання на помилках
   • Дисперсія може збільшитись через складність ансамблю

4. ЧАС НАВЧАННЯ:
   
   • Час навчання лінійно зростає з кількістю n_estimators
   • AdaBoost з n=50 потребує ~50x більше часу ніж одне дерево
   • Це компроміс між точністю та обчислювальними витратами

5. ЗАГАЛЬНИЙ ВИСНОВОК:
   
   AdaBoostRegressor {"покращує" if ada_final_metrics['R2'] > base_final_metrics['R2'] else "не покращує"} 
   якість прогнозування порівняно з базовим деревом рішень.
   Ансамблевий підхід особливо ефективний для даних з нелінійними
   залежностями та шумом, що характерно для фінансових часових рядів.
"""

print(conclusions)

# Збереження результатів
with open('results_summary.txt', 'w', encoding='utf-8') as f:
    f.write("=" * 70 + "\n")
    f.write("ЛАБОРАТОРНА РОБОТА: AdaBoostRegressor\n")
    f.write("Датасет: Google Stock Price\n")
    f.write("=" * 70 + "\n\n")
    f.write(f"Найкращі гіперпараметри:\n")
    f.write(f"  n_estimators: {best_params['n_estimators']}\n")
    f.write(f"  learning_rate: {best_params['learning_rate']}\n")
    f.write(f"  loss: {best_params['loss']}\n\n")
    f.write(f"Тестові метрики:\n")
    f.write(f"  AdaBoost R²: {ada_final_metrics['R2']:.4f}\n")
    f.write(f"  Базова R²:   {base_final_metrics['R2']:.4f}\n\n")
    f.write(conclusions)

print("\nРезультати збережено: results_summary.txt")
print("Усі графіки збережено як PNG файли (1-8)")
print("\n" + "=" * 70)
print("ЛАБОРАТОРНА РОБОТА ЗАВЕРШЕНА УСПІШНО")
print("=" * 70)