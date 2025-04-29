import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

np.random.seed(42)
num_samples = 100

data = pd.DataFrame({
    'GrLivArea': np.random.randint(800, 4000, num_samples),
    'BedroomAbvGr': np.random.randint(1, 6, num_samples),
    'TotRmsAbvGrd': np.random.randint(4, 10, num_samples),
})

data['SalePrice'] = (
    data['GrLivArea'] * 120 + 
    data['BedroomAbvGr'] * 10000 + 
    data['TotRmsAbvGrd'] * 5000 + 
    np.random.normal(0, 10000, num_samples)
)

X = data[['GrLivArea', 'BedroomAbvGr', 'TotRmsAbvGrd']]
y = data['SalePrice']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("✅ Linear Regression Model Evaluation:")
print(f"Mean Squared Error: {mse:.2f}")
print(f"R² Score: {r2:.2f}")

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.7, color='blue')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')  # Perfect prediction line
plt.xlabel("Actual Sale Price")
plt.ylabel("Predicted Sale Price")
plt.title("Actual vs Predicted House Prices")
plt.grid(True)
plt.tight_layout()
plt.show()