import pandas as pd

from tensorflow.keras.datasets import boston_housing

(x_train, y_train), (x_test, y_test) = boston_housing.load_data()

print(x_train.shape)
print(y_train.shape)

classes = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT']
print(classes)

data = pd.DataFrame(x_train, columns=classes)
print(data.head())


data['MEDV'] = pd.Series(data=y_train)
print(data.head())
print(data.describe())
