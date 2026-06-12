# 📈 Linear vs Logistic Regression

A simple, visual demonstration comparing **Linear Regression** and **Logistic Regression** on a small 1D binary classification dataset — showing why logistic regression's sigmoid curve fits classification problems better than a straight line. 🟢🔴

## ✨ Features

- 🧮 Fits both a **Logistic Regression** and a **Linear Regression** model on the same data
- ✅ Compares **accuracy** of both models (using a 0.5 threshold for the linear model)
- 📉 Plots the fitted **sigmoid curve** (logistic) and **fitted line** (linear) against the actual data points

## 📂 Project Structure

```
└── linear_vs_logistic_regression.py
```

## 🚀 Getting Started

### Requirements

```bash
pip install numpy matplotlib scikit-learn
```

### Run

```bash
python linear_vs_logistic_regression.py
```

## 🛠️ How It Works

1. A small 1D dataset with two classes (`0` and `1`) is defined manually.
2. **Logistic Regression** is fit on the data, and predicted probabilities/labels are computed.
3. **Linear Regression** is fit on the same data, and its continuous output is converted to binary predictions using a **0.5 threshold**.
4. **Accuracy** of both models is printed for comparison.
5. A plot shows:
   - 🔴 Class 0 points
   - 🔵 Class 1 points
   - 🟢 The fitted sigmoid curve from logistic regression
   - 🟡 The fitted line from linear regression

## 📈 Output

```
Logistic Regression Accuracy: ...
Linear Regression Accuracy: ...
```

Plus a plot comparing the two fitted curves against the data.

## 📝 Notes

- This example highlights a classic issue with using linear regression for classification: the fitted line can produce probabilities outside the [0, 1] range and is sensitive to outliers (e.g. the point at `x=20`), while logistic regression's sigmoid stays bounded and adapts better to the class separation.

## 👤 Author
Morteza Pazhoum — @MortezaPZ

K.N. Toosi University of Technology — Computer Science
