import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor

# df = pd.DataFrame(
#     {
#         'A' : [1, 2, np.nan, 4],
#         'B' : [np.nan, 12, 3, 4],
#         'C' : [1, 2, 3, 4]
#     }
# )

# 1 - 1
# for i in ('A','B','C'):
#     mean = df[i].mean()
#     df[i].fillna(mean,inplace = True)
# print(df)

# 1 - 2
# i = SimpleImputer(strategy = 'mean')
# df[['A','B']]=i.fit_transform(df[['A','B']])
# print(df)

# 1 - 3
# df [['A','B']]=df[['A','B']].fillna(df[['A','B']].mean())
# print(df)

titanic = sns.load_dataset('titanic')
# print(type(titanic))
#print(titanic.info())
print(titanic.head())
X = titanic[['age']].dropna()
y = titanic.loc[X.index, 'survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train,y_train)
model2 = KNeighborsRegressor()
model2.fit(X_train,y_train)

def evaluate_model(model, X, y, model_name):
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    print(f"{model_name}:")
    print(f"  MSE: {mse:.4f}")
    print(f"  R2: {r2:.4f}")
    print(y_pred)

evaluate_model(model, X_test, y_test, "Linear Regression")
evaluate_model(model2, X_test, y_test, "KNN Regressor")