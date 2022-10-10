#!/usr/bin/env python
# -*- coding=utf-8 -*-
__author__ = "柯博文老師 Powen Ko, www.powenko.com"
import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import time

# 製造 data (共200筆)
np.random.seed(int(time.time()))
num=100
X = np.linspace(-4,4,num)   # 100 筆資料 -4~4
Y = 0.1*np.sin(X)

# 建立 trainig 與 testing data

x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.1)

# 建立 neural network from the first layer to last layer

model = tf.keras.models.Sequential([
   tf.keras.layers.Dense(units=100,
                         activation=tf.nn.tanh,
                         input_dim=1),
   tf.keras.layers.Dense(units=100
                         , activation=tf.nn.tanh
                         ),
   tf.keras.layers.Dense(units=1
                         , activation=tf.nn.tanh
                         ),
])

# 除了第一層以外，定義第二層以上時，不需要定義 input dimension，因為第二層 input 就是第一層的 input

# 開始搭建 model
# mse = mean square error
# sgd = stochastic gradient descent
model.compile(loss='mse',
              #optimizer='sgd',
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              metrics=['acc'])


# training
print("start training")
for step in range(20000):
    cost = model.train_on_batch(x_train, y_train)  #
    if step % 200 == 0:
        #print("train cost{}".format(cost))
        W, b = model.layers[0].get_weights()
        print("step{} Weights = {}, bias = {} train cost{}".format(step,W, b, cost))
        plt.cla()
        # 畫出 data
        plt.scatter(X, Y)
        #X_test2=[-1,1]
        y_pred2 = model.predict(X)  # Y predict
        plt.scatter(X, y_pred2, color='blue')
        plt.text(0, -0.05, 'epoch: %d  ,cost=%.7f '
                 % (step,cost[0]), fontdict={'size': 10, 'color': 'red'})
        plt.pause(0.01)

        #  如果 loss cost  就提早離開
        if (cost[0]<0.000006):
            break
plt.show()


