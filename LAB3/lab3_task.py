import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
"""
from sklearn.datasets import load_iris
iris = load_iris()

data = iris.data
features = iris.feature_names

features = ['sepal_length', 'sepal_width',
            'petal_length', 'petal_width']

df2 = pd.DataFrame(data, columns=features)
"""

df = pd.read_csv('iris_flower.csv')
sns.scatterplot(data=df, x='sepal_length', 
                y='sepal_width', hue='species',
                markers={'Iris-virginica':'s', 'Iris-setosa':'X', 'Iris-versicolor':'+'})
plt.title('Sepal (Length vs Width)')
plt.show()







