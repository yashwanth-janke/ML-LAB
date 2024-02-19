import pandas as pd
import numpy as np

#df = pd.read_csv('Melbourne_housing_FULL.txt')
#df = df.to_csv('Melbourne_housing_FULL.csv')


df = pd.read_csv('Melbourne_housing_FULL.csv')
print(df.info())
print(df.shape)

print(df.isnull())
print(df.notnull())
print(df.isna())

#should not perform this,  as we lost so many instances
dropped = df.dropna()


unique_val = df['Price'].unique()


duplicates = df.drop_duplicates()

fill_mean = df['Price'].fillna(df['Price'].mean())

fill_median = df['Price'].fillna(df['Price'].median())

fill_mode = df['Price'].fillna(df['Price'].mode().iloc[0])

filled = fill_mean + fill_median + fill_mode

ffil = df['Price'].fillna(method='ffill')
bfil = df['Price'].fillna(method='bfill') 

lin = df['Price'].interpolate(method='linear')
quad = df['Price'].interpolate(method='quadratic')