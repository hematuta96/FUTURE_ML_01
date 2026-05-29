import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

# Load dataset
data = pd.read_csv("dataset/train.csv")

# Convert date
data['Order Date'] = pd.to_datetime(
    data['Order Date'],
    format='%d/%m/%Y'
)

# Daily sales
sales_data = data.groupby('Order Date')['Sales'].sum().reset_index()

# Numerical feature
sales_data['Day'] = np.arange(len(sales_data))

X = sales_data[['Day']]
y = sales_data['Sales']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Test prediction
y_pred = model.predict(X_test)

# Metrics
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print("\nMODEL PERFORMANCE")
print("---------------------")
print(f"R² Score : {r2:.4f}")
print(f"MAE      : {mae:.2f}")

# Future forecast
future_days = pd.DataFrame({
    'Day': np.arange(len(sales_data), len(sales_data)+30)
})

future_sales = model.predict(future_days)

print("\nNEXT 30 DAYS FORECAST")
print("---------------------")

for i, sale in enumerate(future_sales[:10], start=1):
    print(f"Day {i}: ₹{sale:.2f}")

# Graph 1 - Historical Sales
plt.figure(figsize=(12,6))
plt.plot(
    sales_data['Order Date'],
    sales_data['Sales']
)
plt.title("Historical Sales Trend")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.grid(True)
plt.show()

# Graph 2 - Future Forecast
plt.figure(figsize=(12,6))
plt.plot(
    range(len(future_sales)),
    future_sales
)
plt.title("Next 30 Days Sales Forecast")
plt.xlabel("Future Days")
plt.ylabel("Predicted Sales")
plt.grid(True)
plt.show()