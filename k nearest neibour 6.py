
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

url = "https://raw.githubusercontent.com/uiuc-cse/data-fa14/gh-pages/data/iris.csv"
data = pd.read_csv(url)

X = data.drop('species', axis=1)
y = data['species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

dt_model = DecisionTreeClassifier(criterion='entropy', random_state=42)
knn_model = KNeighborsClassifier(n_neighbors=5)

dt_model.fit(X_train, y_train)
knn_model.fit(X_train_scaled, y_train)

y_pred_dt = dt_model.predict(X_test)
y_pred_knn = knn_model.predict(X_test_scaled)

print("\nDecision Tree Results:")
print("Accuracy:", round(accuracy_score(y_test, y_pred_dt)*100, 2), "%")
print(confusion_matrix(y_test, y_pred_dt))
print(classification_report(y_test, y_pred_dt))

print("\nKNN Results:")
print("Accuracy:", round(accuracy_score(y_test, y_pred_knn)*100, 2), "%")
print(confusion_matrix(y_test, y_pred_knn))
print(classification_report(y_test, y_pred_knn))

