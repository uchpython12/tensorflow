import tensorflow as tf
import pandas as pd
from tensorflow.keras.datasets import boston_housing
from sklearn.metrics import mean_squared_error, mean_absolute_error
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
(x_train, y_train), (x_test, y_test) = boston_housing.load_data()

print(x_train.shape)
print(y_train.shape)

classes = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT']
data = pd.DataFrame(x_train, columns=classes)
print(data.head())


data['MEDV'] = pd.Series(data=y_train)
print(data.head())
print(data.describe()) #get some basic stats on the dataset

from sklearn import preprocessing
# 標準化
scaler = preprocessing.MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)

model =  tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(512, activation='tanh', input_shape=[x_train.shape[1]]))
model.add(tf.keras.layers.Dense(512, activation='tanh'))
model.add(tf.keras.layers.Dense(1))


model.compile(loss='mse',optimizer='sgd', metrics=['mae'])

history=model.fit(x_train, y_train,
          epochs=2000,
          batch_size=len(y_train))

#保存模型架構
with open("model_Boston.json", "w") as json_file:
   json_file.write(model.to_json())
#保存模型權重
model.save_weights("model_Boston.h5")


# testing
print("start testing")
cost = model.evaluate(x_test, y_test)
print("test cost: {}".format(cost))

Y_pred2 = model.predict(x_test)  # Y predict

print(Y_pred2[:10])
print(y_test[:10])
# 印出測試的結果
Y_pred = model.predict(x_test)
print("預測:",Y_pred )
print("實際:",y_test)
print('MAE:', mean_absolute_error(Y_pred, y_test))
print('MSE:', mean_squared_error(Y_pred, y_test))

