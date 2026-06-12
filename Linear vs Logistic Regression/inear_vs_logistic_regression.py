import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score

X = np.array([[-4.5], [-4], [-3.5], [-3], [-2.5], [-2], [-1.5], [-1], [-0.5], [20]])
y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

model = LogisticRegression(fit_intercept=True)
model.fit(X, y)

model_ = LinearRegression(fit_intercept=True)
model_.fit(X, y)

y_prob = model.predict_proba(X)[:, 1]  
y_pred_logistic = model.predict(X)    

y_pred_linear = model_.predict(X)     
y_pred_linear_binary = (y_pred_linear >= 0.5).astype(int)  

logreg_accuracy = accuracy_score(y, y_pred_logistic)
linreg_accuracy = accuracy_score(y, y_pred_linear_binary)

print(f"Logistic Regression Accuracy: {logreg_accuracy}")
print(f"Linear Regression Accuracy: {linreg_accuracy}")

plt.figure(figsize=(10, 6))
plt.scatter(X[y == 0], y[y == 0], color='red', label='Class 0')
plt.scatter(X[y == 1], y[y == 1], color='blue', label='Class 1')

x_vals = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
y_vals = model.predict_proba(x_vals)[:, 1]
plt.plot(x_vals, y_vals, color='green', label='Fitted Sigmoid (Logistic)')

y__vals = model_.predict(x_vals)
plt.plot(x_vals, y__vals, color='yellow', label='Fitted Line (Linear)')

plt.xlabel('Feature')
plt.ylabel('Probability of Class 1')
plt.title('Logistic Regression and Linear Regression with One Feature')
plt.legend()
plt.grid(True)
plt.show()
