import pandas as pd
data = [1, 7, 5, 2, 8, 3, 6, 4]
bins = [0, 3, 6, 9]
labels = ['low', 'mid', 'high']
cat =  pd.cut(data, bins = bins, right = True, labels = labels)
print(cat)
print(type(cat))