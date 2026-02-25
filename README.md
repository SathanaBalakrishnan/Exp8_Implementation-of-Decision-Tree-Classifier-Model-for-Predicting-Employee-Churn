# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1. Import pandas
2. Import Decision tree classifier
3. Fit the data in the model
4. Find the accuracy score 

## Program:

```
import pandas as pd
data = pd.read_csv("Employee.csv")
print("data.head():")
print(data.head())
print("data.info():")
print(data.info())
print("isnull() and sum():")
print(data.isnull().sum())
print("data value counts():")
print(data["left"].value_counts())
data.columns = data.columns.str.strip()
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data["salary"] = le.fit_transform(data["salary"])
data["Departments"] = le.fit_transform(data["Departments"])
print("data.head() after encoding:")
print(data.head())
x = data[[
    "satisfaction_level",
    "last_evaluation",
    "number_project",
    "average_montly_hours",
    "time_spend_company",
    "Work_accident",
    "promotion_last_5years",
    "Departments",
    "salary"
]]

print("x.head():")
print(x.head())
y = data["left"]
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train, y_train)
y_pred = dt.predict(x_test)
from sklearn import metrics
accuracy = metrics.accuracy_score(y_test, y_pred)
print("Accuracy value:", accuracy)
print("Data Prediction:")
print(dt.predict([[0.5, 0.8, 9, 260, 6, 0, 0, 2, 1]]))
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
plt.figure(figsize=(12,8))
plot_tree(dt, feature_names=x.columns, class_names=['Stayed', 'Left'], filled=True)
plt.show()

```

## Output:

![decision tree classifier model](sam.png)

<img width="992" height="462" alt="image" src="https://github.com/user-attachments/assets/93c069e0-30d4-4213-adc5-4eb06512d97d" />

<img width="906" height="345" alt="image" src="https://github.com/user-attachments/assets/f99a3f95-ea52-4526-84e3-27bf954f572e" />

<img width="862" height="412" alt="image" src="https://github.com/user-attachments/assets/bc69093a-6ed1-41b2-bdfb-d82d893f878e" />

<img width="1000" height="485" alt="image" src="https://github.com/user-attachments/assets/0f6f6f0f-66c2-4886-946f-9e04da143d71" />

<img width="975" height="481" alt="image" src="https://github.com/user-attachments/assets/9f73ca03-8089-4e6d-bb03-c60af6fd5224" />

<img width="1686" height="112" alt="image" src="https://github.com/user-attachments/assets/a589a2fe-9cf7-48fb-bfa0-2b98a339607a" />

<img width="1212" height="783" alt="image" src="https://github.com/user-attachments/assets/0f72ed77-07bb-4e4a-8227-1dae32bd175a" />


## Result:

Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
