from sklearn.datasets import load_iris
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


iris = load_iris()

data = iris.data

feature_names = iris.feature_names

X = pd.DataFrame(data, columns=feature_names)
y = iris.target

minimum = 0
maximum = 1

scaler = MinMaxScaler(feature_range=(minimum, maximum))
X_norm = scaler.fit_transform(X)
      
X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.3, random_state=42)

knn_classifier = KNeighborsClassifier(n_neighbors=4)

knn_classifier.fit(X_train, y_train)

y_pred = knn_classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred, normalize=False)

cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True, cmap='crest', vmax=38, fmt='.1f', linewidth=0, xticklabels=['setosa' , 'versicolor', 'virginica'], yticklabels=['setosa' , 'versicolor', 'virginica'])
plt.ylabel('Prediction',fontsize=13)
plt.xlabel('Actual',fontsize=13)
plt.title('Confusion Matrix',fontsize=17)
plt.show()






