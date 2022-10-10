from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import Myfun
Myfun.pyplot_中文()

X = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
Y = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])



x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.2)

# 建立模型
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(units=1,
        input_dim=1))


model.compile(loss='mse', optimizer='sgd', metrics=['acc'])

# 保存模型架構
with open("model.json", "w") as json_file:
    json_file.write(model.to_json())

lastCost=1.0
for step in range(20000):
    cost = model.train_on_batch(x_train, y_train)  #
    if step % 200 == 0:
        W, b = model.layers[0].get_weights()
        print("step",step," Weights = ",W,", bias =",b," train cost",cost)
        plt.cla()
        # 畫出 data
        plt.scatter(X, Y)
        X_test2=[0,1]
        Y_pred2 = model.predict(X_test2)  # Y predict
        plt.plot(X_test2, Y_pred2,"r-")
        plt.text(0, 1, 'epoch:%d ,權重(Weights)=4%.f ,偏移(bias)=%.4f ,準確率(cost)=%.9f ' % (step, W, b,cost[0]),
                 fontdict={'size': 10, 'color': 'red'})
        plt.axis([-0.2,1.2,-0.2,1.2])   # 定義畫面的XY位置 [xmin, xmax, ymin, ymax]
        plt.pause(0.005)

        #  如果答案比較好，就儲存下來
        if(cost[0]<lastCost):
            lastCost=cost[0]
            # 保存模型權重
            model.save_weights("model.h5")
        #  如果 loss cost  就提早離開
        if (lastCost<0.00000001):
            break

plt.show()


