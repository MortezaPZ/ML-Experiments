import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score

def plot_2d(data_2d, labels, title):
    plt.figure(figsize=(8, 6))
    plt.scatter(data_2d[:, 0], data_2d[:, 1], c=labels, cmap='viridis', s=30)
    plt.title(title)
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.show()

def plot_3d(data_3d, labels, title):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data_3d[:, 0], data_3d[:, 1], data_3d[:, 2], c=labels, cmap='viridis', s=30)
    ax.set_title(title)
    plt.show()

data = load_digits().data
labels_true = load_digits().target

scaled_data = StandardScaler().fit_transform(data)

data_2d = PCA(n_components=2).fit_transform(scaled_data)
data_3d = PCA(n_components=3).fit_transform(scaled_data)

print("\nRunning K-Means...")
ks = range(1, 21)
inertias = []

for k in ks:
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(scaled_data)
    inertias.append(km.inertia_)

plt.figure(figsize=(8, 4))
plt.plot(ks, inertias, marker='o')
plt.title('Elbow Method')
plt.xlabel('Clusters (K)')
plt.ylabel('Inertia')
plt.grid(True)
plt.show()

km_model = KMeans(n_clusters=20, random_state=42)
km_labels = km_model.fit_predict(scaled_data)

plot_2d(data_2d, km_labels, 'K-Means (2D)')
plot_3d(data_3d, km_labels, 'K-Means (3D)')

km_ari = adjusted_rand_score(labels_true, km_labels)
print(f'K-Means ARI: {km_ari:.4f}')

print("\nRunning DBSCAN...")
eps_list = [2.0, 5.0, 10.0]
min_samples_list = [5, 10, 15]

results = []
for eps in eps_list:
    for min_samples in min_samples_list:
        db_model = DBSCAN(eps=eps, min_samples=min_samples)
        db_labels = db_model.fit_predict(scaled_data)

        clusters = len(set(db_labels)) - (1 if -1 in db_labels else 0)
        noise = list(db_labels).count(-1)

        ari = adjusted_rand_score(labels_true, db_labels)
        results.append((eps, min_samples, clusters, noise, ari))

        print(f"eps={eps}, min_samples={min_samples} => Clusters: {clusters}, Noise: {noise}, ARI: {ari:.4f}")

        plot_2d(data_2d, db_labels, f'DBSCAN (2D) - eps={eps}, min_samples={min_samples}')
        plot_3d(data_3d, db_labels, f'DBSCAN (3D) - eps={eps}, min_samples={min_samples}')
