import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.compose import ColumnTransformer


mpg = sns.load_dataset('mpg')
mpg_dropped = mpg.dropna()

X = mpg_dropped.drop(columns=['name','mpg'])
y = mpg_dropped['mpg']

ct = ColumnTransformer(
    [('onehot', OneHotEncoder(handle_unknown='ignore'), ['origin'])],
    remainder='passthrough'
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = ct.fit_transform(X_train)
X_test = ct.transform(X_test)


scaler = StandardScaler(with_mean=False)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


def evaluate_model(model, X, y, model_name):
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    print(f"{model_name}:")
    print(f"  MSE: {mse:.4f}")
    print(f"  R2: {r2:.4f}")

    plt.figure(figsize=(10, 10))
    plt.scatter(y, y_pred, alpha=0.5)
    min_val = min(y.min(), y_pred.min())
    max_val = max(y.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()


model = LinearRegression()
model.fit(X_train_scaled, y_train)

evaluate_model(model, X_test_scaled, y_test, "Linear")


