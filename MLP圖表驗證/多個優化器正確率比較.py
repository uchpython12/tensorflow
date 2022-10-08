"""
 多個優化器(optimizers)進行比較
"Adadelta 訓練時的正確率"
"Adam 訓練時的正確率"
"Nadam 訓練時的正確率"
"SGD 訓練時的正確率"
"Adamax 訓練時的正確率"
"Adagrad 訓練時的正確率"
"RMSprop 訓練時的正確率"
"""
from sklearn import datasets
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np


iris = datasets.load_iris()

category=3
dim=4
train_x , test_x , train_y , test_y= train_test_split(iris.data,iris.target,test_size=0.2)
train_y2=tf.keras.utils.to_categorical(train_y, num_classes=(category))
test_y2=tf.keras.utils.to_categorical(test_y, num_classes=(category))




def AI_MLP(opti1,train_x , test_x , train_y , test_y,train_y2 , test_y2):
    # 建立模型
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(units=100,
                                    activation=tf.nn.relu,
                                    input_dim=dim))
    model.add(tf.keras.layers.Dense(units=100,
                                    activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(units=100,
                                    activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(units=category,
                                    activation=tf.nn.softmax))
    """
    opti1 = tf.keras.optimizers.Adadelta(lr=0.0001)  # Adadelta
    opti1 = tf.keras.optimizers.Adam(lr=0.0001)  # 使用Adam 移動 0.001  #  內定值 learning_rate=0.001,
    opti1 = tf.keras.optimizers.Nadam(lr=0.0001)
    opti1 = tf.keras.optimizers.SGD(lr=0.0001)  # 梯度下降
    """
    model.compile(optimizer=opti1,
                  loss=tf.keras.losses.categorical_crossentropy,
                  metrics=['accuracy'])
    history=model.fit(train_x, train_y2,
              epochs=200,
              batch_size=30,
              verbose=0,  # 訓練時顯示的訊息的狀態，0 無顯示、1 進度、2 詳細
              validation_split=0.3  # 如是 0.3，在訓練時會拿 30% 的數據自行驗證數據)
              )
    # 測試
    score = model.evaluate(test_x, test_y2, batch_size=128)
    print("score:", score)
    predict = model.predict(test_x)
    print("Ans:", np.argmax(predict, axis=-1))
    return model, history

model, history_Adadelta=AI_MLP(tf.keras.optimizers.Adadelta(learning_rate=0.0001),train_x , test_x , train_y , test_y,train_y2 , test_y2)
model, history_Adam=AI_MLP(tf.keras.optimizers.Adam(learning_rate=0.0001),train_x , test_x , train_y , test_y,train_y2 , test_y2)
model, history_Nadam=AI_MLP(tf.keras.optimizers.Nadam(learning_rate=0.0001),train_x , test_x , train_y , test_y,train_y2 , test_y2)
model, history_SGD=AI_MLP(tf.keras.optimizers.SGD(learning_rate=0.0001),train_x , test_x , train_y , test_y,train_y2 , test_y2)

model, history_Adamax=AI_MLP(tf.keras.optimizers.Adamax(learning_rate=0.0001),train_x , test_x , train_y , test_y,train_y2 , test_y2)
model, history_Adagrad=AI_MLP(tf.keras.optimizers.Adagrad(learning_rate=0.0001),train_x , test_x , train_y , test_y,train_y2 , test_y2)
model, history_RMSprop=AI_MLP(tf.keras.optimizers.RMSprop(learning_rate=0.0001),train_x , test_x , train_y , test_y,train_y2 , test_y2)

#
import matplotlib.pyplot as plt
import myfun
myfun.pyplot_中文()

plt.plot(history_Adadelta.history['accuracy'],label="Adadelta 訓練時的正確率")
plt.plot(history_Adam.history['accuracy'],label="Adam 訓練時的正確率")
plt.plot(history_Nadam.history['accuracy'],label="Nadam 訓練時的正確率")
plt.plot(history_SGD.history['accuracy'],label="SGD 訓練時的正確率")
plt.plot(history_Adamax.history['accuracy'],label="Adamax 訓練時的正確率")
plt.plot(history_Adagrad.history['accuracy'],label="Adagrad 訓練時的正確率")
plt.plot(history_RMSprop.history['accuracy'],label="RMSprop 訓練時的正確率")

# val_loss: 0.0852 - val_accuracy

plt.title('model accuracy')
plt.ylabel('accuracy , loss, validation accuracy, validation loss')
plt.xlabel('epoch')
plt.legend(loc='lower left')
plt.show()