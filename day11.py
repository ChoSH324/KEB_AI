import numpy as np
import pandas as pd

df = pd.DataFrame(
    {"a" :[4, 7 , 10],
     "b" : [5, 88, 11],
     "c" : [16, 9, 12]},
    index = [1, 2, 3]
    )
# print('df',df)

# print(pd.melt(df))
# df2=(pd.melt(df).rename(columns={'variable':'var','value':'val'}).query('val > 10').sort_values('val'))
# print(df2)

# df3 = df.iloc[1:2]
df3 = df.iloc[:,[0,2]]
print(df3)

df4=df.loc[1:2]
print(df4)

def square(n)->int:
    return n*n


print(df.apply(square))
# print(df.apply(lambda x : x* x))