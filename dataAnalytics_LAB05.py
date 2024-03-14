from sklearn import tree
X = [[0, 0], [1, 1]]
Y = [0, 1]
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, Y)

clf.predict([[2., 2.]])

clf.predict_proba([[2., 2.]])

from sklearn.datasets import load_iris
from sklearn import tree
iris = load_iris()
X, y = iris.data, iris.target
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, y)

tree.plot_tree(clf)

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import tree
import matplotlib.pyplot as plt

file_path = '/content/Physical_examination_dataset.csv'
data = pd.read_csv(file_path)
data = pd.get_dummies(data, columns=['Blood glucose', 'Weight'])
X = data.drop('Diabetic(Yes or No)', axis=1)
Y = data['Diabetic(Yes or No)']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=43)

clf = DecisionTreeClassifier()

clf.fit(X_train, Y_train)
plt.figure(figsize=(15, 10))
tree.plot_tree(clf, feature_names=X.columns, class_names=['No', 'Yes'], filled=True)
plt.show()