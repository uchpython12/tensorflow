import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error
import myfun

classes = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT']

train_x, test_x, train_y, test_y,scalerX,scalerY=\
              myfun.ML_read_dataframe_標準化xy("boston.xlsx",
              classes,
              ['MEDV'])


model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(320, activation='tanh', input_shape=[train_x.shape[1]]))
model.add(tf.keras.layers.Dense(640, activation='tanh'))
model.add(tf.keras.layers.Dense(640, activation='tanh'))
model.add(tf.keras.layers.Dense(1))

# learning_rate = 0.000001
learning_rate = 0.0001
opt1 = tf.keras.optimizers.Nadam(learning_rate=learning_rate)
model.compile(loss='mse', optimizer=opt1, metrics=['mae'])


try:
    with open('model2.h5', 'r') as load_weights:
        # 讀取模型權重
        model.load_weights("model2.h5")
except IOError:
    print("File not exists")

history=model.fit(train_x, train_y,
          epochs=1000,
          batch_size=10000)

#保存模型架構
with open("model2.json", "w") as json_file:
   json_file.write(model.to_json())
#保存模型權重
model.save_weights("model2.h5")



import matplotlib.pyplot as plt

print(history.history.keys())
plt.plot(history.history['mae']) # mean_absolute_error
plt.title('Boston house price')
plt.ylabel('mae')
plt.xlabel('epoch')
plt.legend(['train mae'], loc='upper right')
plt.show()




# testing
print("start testing")
cost = model.evaluate(test_x, test_y)
print("test cost: {}".format(cost))



# 印出測試的結果
Y_pred = model.predict(test_x)
test_y3=myfun.ML_標準化1_還原Y(test_y, scalerY)
Y_pred3=myfun.ML_標準化1_還原Y(Y_pred, scalerY)

print("預測:",Y_pred )
print("實際:",test_y)
print('MAE:', mean_absolute_error(Y_pred, test_y))
print('MSE:', mean_squared_error(Y_pred, test_y))



print("預測標準化還原的答案:",Y_pred3 )
print("實際標準化還原的答案::",test_y3)



