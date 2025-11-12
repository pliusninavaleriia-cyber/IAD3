
import time
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import make_moons, make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import SpectralClustering, KMeans, MiniBatchKMeans, Birch, DBSCAN
from sklearn.metrics import (
    adjusted_rand_score,
    adjusted_mutual_info_score,
    homogeneity_score,
    completeness_score,
    v_measure_score,
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)
from sklearn.metrics.cluster import contingency_matrix
from sklearn.exceptions import ConvergenceWarning

# ------------------------------
# 0. Настройки
# ------------------------------
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ------------------------------
# 1. Генерация датасетов и нормализация
# ------------------------------
X_moons, y_moons = make_moons(n_samples=2000, noise=0.05, random_state=RANDOM_STATE)
X_blobs, y_blobs = make_blobs(n_samples=2000, n_features=2, centers=4,
                              cluster_std=1.0, center_box=(-10.0, 10.0),
                              shuffle=True, random_state=1)

scaler = StandardScaler()
X_moons_scaled = scaler.fit_transform(X_moons)
X_blobs_scaled = scaler.fit_transform(X_blobs)

# ------------------------------
# 2. Визуализация исходных данных
# ------------------------------
def plot_initial_data():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14,6))
    ax1.scatter(X_moons[:,0], X_moons[:,1], c=y_moons, s=12, cmap='viridis')
    ax1.set_title('Moons - True labels')
    ax2.scatter(X_blobs[:,0], X_blobs[:,1], c=y_blobs, s=12, cmap='viridis')
    ax2.set_title('Blobs - True labels')
    plt.tight_layout()
    plt.show()

plot_initial_data()

# ------------------------------
# 3. Утилиты: безопасный запуск Spectral + таймер
# ------------------------------
def safe_spectral_clustering(X, n_clusters, **kwargs):
    """Wrapper для SpectralClustering с обработкой ошибок и измерением времени.
       Если алгоритм упадёт из-за размера/памяти — вернёт (None, elapsed)."""
    start = time.time()
    try:
        model = SpectralClustering(n_clusters=n_clusters, random_state=RANDOM_STATE, **kwargs)
        labels = model.fit_predict(X)
        return labels, time.time() - start
    except Exception as e:
        print(f"[Spectral ERROR] n_clusters={n_clusters}, params={kwargs} -> {e}")
        return None, time.time() - start

def time_function(func, *args, **kwargs):
    start = time.time()
    res = func(*args, **kwargs)
    return res, time.time() - start

# ------------------------------
# 4. Полный набор метрик (по заданию)
# ------------------------------
def compute_metrics(X, y_true, y_pred):
    """Возвращает словарь всех метрик, требуемых по заданию."""
    if y_pred is None:
        return None
    metrics = {}
    metrics['n_clusters_est'] = int(len(np.unique(y_pred)))
    metrics['ari'] = float(adjusted_rand_score(y_true, y_pred))
    metrics['ami'] = float(adjusted_mutual_info_score(y_true, y_pred))
    metrics['homogeneity'] = float(homogeneity_score(y_true, y_pred))
    metrics['completeness'] = float(completeness_score(y_true, y_pred))
    metrics['v_measure'] = float(v_measure_score(y_true, y_pred))
    if metrics['n_clusters_est'] > 1:
        try:
            metrics['silhouette'] = float(silhouette_score(X, y_pred))
        except Exception:
            metrics['silhouette'] = np.nan
        try:
            metrics['calinski_harabasz'] = float(calinski_harabasz_score(X, y_pred))
        except Exception:
            metrics['calinski_harabasz'] = np.nan
        try:
            metrics['davies_bouldin'] = float(davies_bouldin_score(X, y_pred))
        except Exception:
            metrics['davies_bouldin'] = np.nan
    else:
        metrics['silhouette'] = np.nan
        metrics['calinski_harabasz'] = np.nan
        metrics['davies_bouldin'] = np.nan

    # Contingency matrix
    try:
        metrics['contingency'] = contingency_matrix(y_true, y_pred)
    except Exception:
        metrics['contingency'] = None

    return metrics

def print_metrics(name, metrics):
    if metrics is None:
        print(f"{name}: FAILED (no labels)")
        return
    print(f"\n{name}:")
    print(f"  Estimated clusters: {metrics['n_clusters_est']}")
    print(f"  ARI: {metrics['ari']:.4f}, AMI: {metrics['ami']:.4f}")
    print(f"  Homogeneity: {metrics['homogeneity']:.4f}, Completeness: {metrics['completeness']:.4f}, V-measure: {metrics['v_measure']:.4f}")
    print(f"  Silhouette: {metrics['silhouette']}, Calinski-Harabasz: {metrics['calinski_harabasz']}, Davies-Bouldin: {metrics['davies_bouldin']}")
    if metrics.get('contingency') is not None:
        cm = metrics['contingency']
        print(f"  Contingency matrix shape: {cm.shape}")
        print(f"  Contingency (top-left):\n{cm}")

# ------------------------------
# 5. Базовая кластеризация (Spectral) и визуализация
# ------------------------------
results_rows = []

def run_and_record(name, X_raw, X_scaled, y_true, model_func, plot=True):
    """model_func должен возвращать кортеж (labels, elapsed_time)."""
    (labels, elapsed) = model_func(X_scaled, y_true)
    metrics = compute_metrics(X_scaled, y_true, labels)
    print_metrics(name, metrics)
    row = {'model': name, 'time_s': elapsed}
    if metrics is not None:
        for k,v in metrics.items():
            if k == 'contingency':
                row['contingency_shape'] = None if v is None else v.shape
            else:
                row[k] = v
    results_rows.append(row)

    # Визуализация
    if plot and labels is not None:
        plt.figure(figsize=(5,4))
        plt.scatter(X_raw[:,0], X_raw[:,1], c=labels, s=10, cmap='viridis')
        plt.title(f"{name} (time={elapsed:.3f}s)")
        plt.tight_layout()
        plt.show()
    return labels, elapsed

# Обёртки моделей:
def spectral_wrapper(X_scaled, y_true, n_clusters=2, **kwargs):
    labels, elapsed = safe_spectral_clustering(X_scaled, n_clusters=n_clusters, **kwargs)
    return labels, elapsed

def kmeans_wrapper(X_scaled, y_true, n_clusters=2, **kwargs):
    start = time.time()
    model = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, **kwargs)
    labels = model.fit_predict(X_scaled)
    return labels, time.time() - start

def minibatch_kmeans_wrapper(X_scaled, y_true, n_clusters=2, **kwargs):
    start = time.time()
    model = MiniBatchKMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, **kwargs)
    labels = model.fit_predict(X_scaled)
    return labels, time.time() - start

def birch_wrapper(X_scaled, y_true, n_clusters=2, **kwargs):
    start = time.time()
    model = Birch(n_clusters=n_clusters, **kwargs)
    labels = model.fit_predict(X_scaled)
    return labels, time.time() - start

def dbscan_wrapper(X_scaled, y_true, **kwargs):
    start = time.time()
    model = DBSCAN(**kwargs)
    labels = model.fit_predict(X_scaled)
    return labels, time.time() - start

# ------------------------------
# 6. Запуск основных/альтернативных моделей для MOONS
# ------------------------------
print("\n=== MOONS: основные и альтернативные модели ===")
# Базовый spectral
run_and_record("Spectral (rbf, gamma=1.0) - Moons",
               X_moons, X_moons_scaled, y_moons,
               lambda Xs, yt: spectral_wrapper(Xs, yt, n_clusters=2, affinity='rbf', gamma=1.0))

# Альтернативы 
configs_moons = [
    ("Spectral (rbf, gamma=0.1) - Moons", {'affinity':'rbf','gamma':0.1, 'n_clusters':2}),
    ("Spectral (rbf, gamma=5.0) - Moons", {'affinity':'rbf','gamma':5.0, 'n_clusters':2}),
    ("Spectral (nn, k=10) - Moons", {'affinity':'nearest_neighbors','n_neighbors':10, 'n_clusters':2}),
    ("Spectral (rbf, gamma=1.0, k=3) - Moons", {'affinity':'rbf','gamma':1.0, 'n_clusters':3}),
    ("KMeans (k=2) - Moons", {'k':2}),
    ("MiniBatchKMeans (k=2) - Moons", {'k':2}),
    ("Birch (k=2) - Moons", {'k':2}),
    ("DBSCAN (eps=0.2) - Moons", {'eps':0.2, 'min_samples':5})
]

for name, params in configs_moons:
    if name.startswith("Spectral"):
        n_clusters = params.pop('n_clusters', 2)
        run_and_record(name, X_moons, X_moons_scaled, y_moons,
                       lambda Xs, yt: spectral_wrapper(Xs, yt, n_clusters=n_clusters, **params))
    elif name.startswith("KMeans"):
        run_and_record(name, X_moons, X_moons_scaled, y_moons,
                       lambda Xs, yt: kmeans_wrapper(Xs, yt, n_clusters=params['k']))
    elif name.startswith("MiniBatchKMeans"):
        run_and_record(name, X_moons, X_moons_scaled, y_moons,
                       lambda Xs, yt: minibatch_kmeans_wrapper(Xs, yt, n_clusters=params['k'], batch_size=256))
    elif name.startswith("Birch"):
        run_and_record(name, X_moons, X_moons_scaled, y_moons,
                       lambda Xs, yt: birch_wrapper(Xs, yt, n_clusters=params['k']))
    elif name.startswith("DBSCAN"):
        run_and_record(name, X_moons, X_moons_scaled, y_moons,
                       lambda Xs, yt: dbscan_wrapper(Xs, yt, eps=params['eps'], min_samples=params.get('min_samples',5)))

# ------------------------------
# 7. Запуск основных/альтернативных моделей для BLOBS
# ------------------------------
print("\n=== BLOBS: основные и альтернативные модели ===")
run_and_record("Spectral (rbf, gamma=1.0) - Blobs",
               X_blobs, X_blobs_scaled, y_blobs,
               lambda Xs, yt: spectral_wrapper(Xs, yt, n_clusters=4, affinity='rbf', gamma=1.0))

configs_blobs = [
    ("Spectral (rbf, gamma=0.1) - Blobs", {'affinity':'rbf','gamma':0.1, 'n_clusters':4}),
    ("Spectral (rbf, gamma=5.0) - Blobs", {'affinity':'rbf','gamma':5.0, 'n_clusters':4}),
    ("Spectral (nn, k=15) - Blobs", {'affinity':'nearest_neighbors','n_neighbors':15, 'n_clusters':4}),
    ("Spectral (rbf, gamma=1.0, k=3) - Blobs", {'affinity':'rbf','gamma':1.0, 'n_clusters':3}),
    ("KMeans (k=4) - Blobs", {'k':4}),
    ("MiniBatchKMeans (k=4) - Blobs", {'k':4}),
    ("Birch (k=4) - Blobs", {'k':4}),
    ("DBSCAN (eps=1.0) - Blobs", {'eps':1.0, 'min_samples':5})
]

for name, params in configs_blobs:
    if name.startswith("Spectral"):
        n_clusters = params.pop('n_clusters', 4)
        run_and_record(name, X_blobs, X_blobs_scaled, y_blobs,
                       lambda Xs, yt: spectral_wrapper(Xs, yt, n_clusters=n_clusters, **params))
    elif name.startswith("KMeans"):
        run_and_record(name, X_blobs, X_blobs_scaled, y_blobs,
                       lambda Xs, yt: kmeans_wrapper(Xs, yt, n_clusters=params['k']))
    elif name.startswith("MiniBatchKMeans"):
        run_and_record(name, X_blobs, X_blobs_scaled, y_blobs,
                       lambda Xs, yt: minibatch_kmeans_wrapper(Xs, yt, n_clusters=params['k'], batch_size=256))
    elif name.startswith("Birch"):
        run_and_record(name, X_blobs, X_blobs_scaled, y_blobs,
                       lambda Xs, yt: birch_wrapper(Xs, yt, n_clusters=params['k']))
    elif name.startswith("DBSCAN"):
        run_and_record(name, X_blobs, X_blobs_scaled, y_blobs,
                       lambda Xs, yt: dbscan_wrapper(Xs, yt, eps=params['eps'], min_samples=params.get('min_samples',5)))

# ------------------------------
# 8. Тест масштабируемости (performance)
# ------------------------------
print("\n=== SCALABILITY TEST (time) ===")
def scalability_test(sizes=[1000, 5000, 10000, 20000]):
    rows = []
    for n in sizes:
        print(f"\n-- Size: {n}")
        X_test, _ = make_blobs(n_samples=n, n_features=2, centers=3, random_state=RANDOM_STATE)
        X_test_scaled = scaler.fit_transform(X_test)

        # Spectral only for small n (<=5000) 
        if n <= 5000:
            _, sp_time = safe_spectral_clustering(X_test_scaled, n_clusters=3, affinity='rbf', gamma=1.0)
        else:
            sp_time = np.nan

        # KMeans
        start = time.time()
        KMeans(n_clusters=3, random_state=RANDOM_STATE).fit(X_test_scaled)
        kmeans_time = time.time() - start

        # MiniBatchKMeans
        start = time.time()
        MiniBatchKMeans(n_clusters=3, random_state=RANDOM_STATE, batch_size=1000).fit(X_test_scaled)
        mb_time = time.time() - start

        # Birch 
        start = time.time()
        Birch(n_clusters=3).fit(X_test_scaled)
        birch_time = time.time() - start

        print(f"Times (s): spectral={sp_time}, kmeans={kmeans_time:.4f}, minibatch={mb_time:.4f}, birch={birch_time:.4f}")
        rows.append({'n': n, 'spectral_time': sp_time, 'kmeans_time': kmeans_time,
                     'minibatch_time': mb_time, 'birch_time': birch_time})
    return pd.DataFrame(rows)

scalability_df = scalability_test([1000, 5000, 10000, 20000])
print("\nScalability results:")
print(scalability_df)

# ------------------------------
# 9. Расширенный анализ стабильности
# ------------------------------
print("\n=== STABILITY ANALYSIS ===")
def stability_analysis(X_scaled, y_true, base_params, n_trials=5):
    """Проводит три теста: удаление ~10%, shuffle, подвыборки 50/70%"""
    n = X_scaled.shape[0]
    stats = {'remove': [], 'shuffle': [], 'subsample': []}
    for t in range(n_trials):
    
        mask = np.random.choice([True, False], size=n, p=[0.1, 0.9])
        X_r = X_scaled[~mask]
        y_r = y_true[~mask]
        if len(np.unique(y_r)) < 1:
            continue
        y_pred_r, _ = safe_spectral_clustering(X_r, n_clusters=len(np.unique(y_r)), **base_params)
        if y_pred_r is not None:
            stats['remove'].append(adjusted_rand_score(y_r, y_pred_r))

        # shuffle
        perm = np.random.permutation(n)
        X_s = X_scaled[perm]
        y_s = y_true[perm]
        y_pred_s, _ = safe_spectral_clustering(X_s, n_clusters=len(np.unique(y_s)), **base_params)
        if y_pred_s is not None:
            stats['shuffle'].append(adjusted_rand_score(y_s, y_pred_s))

        # subsample 50% or 70%
        frac = 0.5 if (t % 2 == 0) else 0.7
        idx = np.random.choice(n, size=int(n*frac), replace=False)
        X_sub = X_scaled[idx]
        y_sub = y_true[idx]
        y_pred_sub, _ = safe_spectral_clustering(X_sub, n_clusters=len(np.unique(y_sub)), **base_params)
        if y_pred_sub is not None:
            stats['subsample'].append(adjusted_rand_score(y_sub, y_pred_sub))
    # печать результатов
    for k, v in stats.items():
        if len(v) > 0:
            print(f"{k}: mean ARI = {np.mean(v):.4f}, std = {np.std(v):.4f}, trials = {len(v)}")
        else:
            print(f"{k}: no valid trials")
    return stats

base_params = {'affinity':'rbf', 'gamma':1.0}
print("\nMoons stability:")
stability_moons = stability_analysis(X_moons_scaled, y_moons, base_params, n_trials=6)
print("\nBlobs stability:")
stability_blobs = stability_analysis(X_blobs_scaled, y_blobs, base_params, n_trials=6)

# ------------------------------
# 10. Сравнение моделей и выбор лучших
# ------------------------------
print("\n=== MODELS COMPARISON ===")
df_results = pd.DataFrame(results_rows)
display_cols = ['model','dataset','n_clusters_est','ari','ami','silhouette','calinski_harabasz','davies_bouldin','time_s']
if 'dataset' not in df_results.columns:
    df_results['dataset'] = df_results['model'].apply(lambda s: 'Moons' if 'Moons' in s else ('Blobs' if 'Blobs' in s else 'Unknown'))

# Упорядочим по dataset и ari
for ds in df_results['dataset'].unique():
    print(f"\nTop models for dataset = {ds}:")
    subset = df_results[df_results['dataset'] == ds].copy()
    subset = subset.sort_values(by='ari', ascending=False)
    print(subset[['model','n_clusters_est','ari','silhouette','calinski_harabasz','davies_bouldin','time_s']].head(8))

# ------------------------------
# 11. Выводы и сохранение результатов
# ------------------------------
print("\n=== FINAL CONCLUSIONS ===")
print("1) SpectralClustering хорошо работает для не-выпуклых структур (например, moons), но плохо масштабируется (матрица смежности/лапласиан).")
print("2) Для blob-подобных данных KMeans / MiniBatchKMeans / Birch часто быстрее и дают хорошие результаты.")
print("3) Для масштабирования до 10k-100k: рекомендуется использовать MiniBatchKMeans, Birch или приближения спектрального метода (Nyström / landmark).")
print("4) Стабильность кластеризации проверялась (удаление 10%, shuffle, подвыборки) — см. выводы выше по среднему ARI и std.")
