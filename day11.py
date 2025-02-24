import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

data_root = "https://github.com/ageron/data/raw/main/"
ls=pd.read_csv(data_root + "lifesat/lifesat.csv")
X = ls[["GDP per capita (USD)"]].values
y = ls[["Life satisfaction"]].values

ls.plot(kind = 'scatter',grid = True, x="GDP per capita (USD)", y ="Life satisfaction")
plt.axis([23_500, 62_500, 4, 9])
plt.show()
model = LinearRegression()
model.fit(X,y)

X_new=[[37_655.2]]
print(model.predict(X_new))