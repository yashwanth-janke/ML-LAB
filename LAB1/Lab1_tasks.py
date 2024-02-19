# importing the necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# reading the data set
df = pd.read_csv('world_population_data.csv')

# for dimensions of a data
print(df.ndim)

# for getting names of columns
print(df.columns)

# displaying data types of features

# for a particular column
print(df['rank'].dtype)

# for all columns in the dataset
print(df.dtypes)

# Computing Statistics

# To get information about all columns in dataset
print(df.info())

#To describe the whole dataset with statistics of each column
print(df.describe()) # Note: the values (25% 50% 75%) represent the first, second, third quantiles

# to display the last few instances or lat few rows of the dataset
last_rows = df.tail(10) # if no value passsed it returns 5 rows by default

# return number of rows and columns of dataset, nothing but shape
x, y = df.shape

# count distinct values
uniq_val = df['continent'].unique()
count_uniq_val = df['continent'].nunique()


#Visualization
df['1980 population'].hist()
plt.title('1980 population')
plt.show()

sns.boxplot(data=df, x='continent', y='rank')
plt.title('1980 population')
plt.show()

plt.scatter(df['continent'], df['rank'])
plt.title('1980 population')

# density
sns.kdeplot(df['rank'])

# Handling missing data
# To remove features with missing values
# IN THIS CASE NO MISSING VALUES
df.dropna()

df['growth rate'].convert_dtypes(infer_objects=True, convert_integer=True)
# filling data in missed slots
#df['growth rate'].fillna(df['growth rate'].mean(), inplace=True)
#df.fillna(df.mean(), inplace = True)


