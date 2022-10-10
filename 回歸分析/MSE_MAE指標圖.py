from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error


X = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
Y = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])


category=1
dim=1
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.2)

# 建立模型
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(units=100,
                                #activation=tf.nn.tanh,
                                input_dim=dim))
model.add(tf.keras.layers.Dense(units=category))


model.compile(loss='mse',
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
              metrics=['mse', 'mae', 'mape',  tf.compat.v1.keras.losses.cosine_proximity])


history=model.fit(x_train, y_train,
       epochs=3000,
       batch_size=len(X))





# testing
cost = model.evaluate(x_test, y_test)
print("準確率 score: ",cost)
weights, biases = model.layers[0].get_weights()
print("權重Weights =" ,weights, "偏移bias = ",biases)

# 印出測試的結果
Y_pred = model.predict(x_test)
print("預測:",Y_pred )
print("實際:",y_test )
print('MAE:', mean_absolute_error(Y_pred, y_test))
print('MSE:', mean_squared_error(Y_pred, y_test))




plt.plot(history.history['mse']) #mean_squared_error
plt.plot(history.history['mae']) # mean_absolute_error
plt.plot(history.history['mape']) # mean_absolute_percentage_error
plt.plot(history.history['cosine_similarity']) # cosine_proximity
plt.title('Regression Metrics')
plt.ylabel('')
plt.xlabel('epoch')
plt.legend(['mean_squared_error','mean_absolute_error',
            'mean_absolute_percentage_error','cosine_proximity'],

           loc='upper right')
plt.savefig("save.jpg")
plt.show()


