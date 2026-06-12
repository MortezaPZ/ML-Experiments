# 🔢 K-Means vs DBSCAN on Digits Dataset

A comparison of two clustering algorithms — **K-Means** and **DBSCAN** — on the scikit-learn **Digits dataset** (handwritten digits 0–9), evaluated using **PCA visualization** and the **Adjusted Rand Index (ARI)**. 🎯

## ✨ Features

- 📏 **Data scaling** with `StandardScaler`
- 📉 **Dimensionality reduction** with PCA (2D and 3D projections for visualization)
- 📊 **Elbow Method** to explore K-Means cluster counts
- 🔵 **K-Means clustering** with `k=20`
- 🌀 **DBSCAN clustering** across multiple `eps` and `min_samples` combinations
- ✅ **Adjusted Rand Index (ARI)** to measure clustering quality against true digit labels

## 📂 Project Structure

```
└── kmeans_vs_dbscan_digits.py
```

## 🚀 Getting Started

### Requirements

```bash
pip install numpy matplotlib scikit-learn
```

### Run

```bash
python kmeans_vs_dbscan_digits.py
```

## 🛠️ How It Works

1. **Preprocessing** — the Digits dataset (1797 samples, 64 features) is standardized with `StandardScaler`.
2. **PCA** — data is projected into 2D and 3D for visualization purposes.
3. **K-Means**:
   - The **Elbow Method** is plotted for `k = 1` to `20` to help choose a reasonable number of clusters.
   - A final K-Means model with `k=20` is fit, and its clusters are visualized in 2D/3D.
   - Clustering quality is measured with **ARI** against the true digit labels.
4. **DBSCAN**:
   - Multiple combinations of `eps` (2.0, 5.0, 10.0) and `min_samples` (5, 10, 15) are tested.
   - For each combination, the number of clusters, noise points, and **ARI** are printed and visualized.

## 📈 Output

- Multiple PCA scatter plots (2D and 3D) for each clustering result
- An elbow plot for K-Means
- Printed **ARI scores** for K-Means and each DBSCAN configuration

## 📝 Notes

- ARI scores allow direct comparison of how well each algorithm's clusters align with the true digit labels (0–9).
- DBSCAN's performance is highly sensitive to `eps` and `min_samples` — the grid search over these parameters helps illustrate this sensitivity.

## 👤 Author
Morteza Pazhoum — @MortezaPZ

K.N. Toosi University of Technology — Computer Science
