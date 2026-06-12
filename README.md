# 🧪 ML Experiments

A collection of small, focused machine learning experiments exploring **clustering** and **regression** concepts with scikit-learn. 🤖

## 📂 Project Structure

```
├── clustering-digits/
│   └── kmeans_vs_dbscan_digits.py        # K-Means vs DBSCAN on the Digits dataset
├── linear-vs-logistic-regression/
│   └── linear_vs_logistic_regression.py  # Linear vs Logistic Regression on a 1D classification task
└── README.md
```

---

## 🔢 Clustering: K-Means vs DBSCAN

Compares **K-Means** and **DBSCAN** clustering on the scikit-learn **Digits dataset**, using PCA for visualization and the **Adjusted Rand Index (ARI)** to evaluate how well each method recovers the true digit groupings.

- 📊 Elbow method for choosing K
- 🎨 2D/3D PCA visualizations of clusters
- 🌀 DBSCAN parameter sweep (`eps`, `min_samples`)

➡️ See [`clustering-digits/README.md`](clustering-digits/README.md) for details.

---

## 📈 Linear vs Logistic Regression

A simple, visual demonstration of why **Logistic Regression** is better suited for classification than **Linear Regression**, using a small 1D dataset with two classes.

- 🟢 Fits both a logistic (sigmoid) and a linear model to the same binary classification data
- 📐 Compares accuracy of both models
- 📉 Plots the fitted sigmoid curve vs. the fitted line alongside the data points

➡️ See [`linear-vs-logistic-regression/README.md`](linear-vs-logistic-regression/README.md) for details.

---

## 🚀 Getting Started

### Requirements

```bash
pip install numpy matplotlib scikit-learn
```

### Run any experiment

```bash
python clustering-digits/kmeans_vs_dbscan_digits.py
python linear-vs-logistic-regression/linear_vs_logistic_regression.py
```
## 📝 Notes

- Each folder is a standalone experiment — no shared code or dependencies between them.
- These scripts are meant for learning and visualization purposes, illustrating core differences between clustering algorithms and regression approaches.
## 👤 Author
Morteza Pazhoum — @MortezaPZ

K.N. Toosi University of Technology — Computer Science
