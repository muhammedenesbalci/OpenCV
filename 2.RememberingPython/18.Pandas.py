import pandas as pd

dic = {"name": ["enes", "furkan", "isa", "selma", "kuzey", "elif"],
       "year": [23, 15, 47, 44, 10, 8],
       "salary": [8000, 0, 20000, 0, 0, 0]}

# Dict ot dataframe
print("Dict ot dataframe-----------------------------")
data = pd.DataFrame(dic)
print(data)

# First rows
print("First rows---------------------------------")
print(data.head(2))

# Column names
print("Column names-------------------------------")
print(data.columns)

# Info about datas
print("Info about datas------------------------------")
print(data.info())

# Statical Infos
print("Statical Infos-------------------------------")
print(data.describe())

# Get specific column
print("Get specific column----------------------------")
print(data["salary"])
print(data.loc[:, "name"])

# Get row
print("Get row-------------------------------------")
print(data.iloc[0])
print(data.iloc[0, 0:2:1])
print(data.iloc[0:2])

# Adding new Column
print("Adding new Column--------------------------")
data["City"] = ["Ankara", "Istanbul", "Istanbul", "Istanbul", "Istanbul", "Istanbul"]
print(data)

# Using Loc [range of row, range of column] last index including (location)
print("Using Loc-----------------------------")
print(data.loc[0])
print(data.loc[0, "City"])
print(data.loc[0:2, "name":"salary"])

# Specific indexes with loc
print("Specific indexes with loc---------------------------------")
print(data.loc[[0, 3], ["name", "salary"]])

# Reversing
print("Reversing-----------------------------------------")
print(data.loc[::-1, :])
print(data.loc[:, ::-1])
print(data.loc[::-1, ::-1])

# Using iloc [range of row, range of column] (index location) last index not including
print("Using iloc-----------------------------------")
print(data.iloc[:, 2])
print(data.iloc[0:2:1, :])

# Specific indexes with iloc
print("Specific indexes with iloc-----------------------------------")
print(data.iloc[[1,2], [2,3]])

# Filtering
print("Filtering-------------------------")
filter1 = data.loc[:, "year"] >= 23
print(data[filter1])

# mean
print("Mean-------------------------")
mean_data_year =data.loc[:, "year"].mean()
print(mean_data_year)

# Combining datasets
print("Combining datasets-------------------------")
dic1 = {"name": ["enes", "furkan", "isa", "selma", "kuzey", "elif"],
       "year": [23, 15, 47, 44, 10, 8],
       "salary": [8000, 0, 20000, 0, 0, 0]}

dic2 = {"name": ["murat", "nihat", "zübeyde", "izem", "ilayda"],
       "year": [23, 48, 44, 12, 20],
       "salary": [8000, 30000, 12000, 0, 0]}

data1=pd.DataFrame(dic1)
data2 = pd.DataFrame(dic2)

print(data1)
print(data2)

# Vertical combining
print("Vertical combining-------------------------")
data_vertical = pd.concat([data1, data2], axis=0)
print(data_vertical)

# Horizontal combining
print("Horizontal combining-------------------------------------")
data_horizontal = pd.concat([data1, data2], axis=1)
print(data_horizontal)





