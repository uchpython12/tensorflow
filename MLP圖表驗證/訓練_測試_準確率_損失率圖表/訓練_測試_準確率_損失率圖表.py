"""損失下降，準確度上升,透過此方法來評估模型的有效性
評估意味著確定模型如何有效地進行預測
但如果在訓練的過程中準確度都沒有上升
代表類神經的模型或者是數據其中有一個有問題
需要進行調整"""

from sklearn import datasets
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np


iris = datasets.load_iris()

category=3
dim=4
x_train , x_test , y_train , y_test = train_test_split(iris.data,iris.target,test_size=0.2)
y_train2=tf.keras.utils.to_categorical(y_train, num_classes=(category))
y_test2=tf.keras.utils.to_categorical(y_test, num_classes=(category))

print("x_train[:4]",x_train[:4])
print("y_train[:4]",y_train[:4])
print("y_train2[:4]",y_train2[:4])

# 建立模型
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(units=10,
    activation=tf.nn.relu,
    input_dim=dim))
model.add(tf.keras.layers.Dense(units=10,
    activation=tf.nn.relu ))
model.add(tf.keras.layers.Dense(units=category,
    activation=tf.nn.softmax ))
model.compile(optimizer='adam',
    loss=tf.keras.losses.categorical_crossentropy,
    metrics=['accuracy'])
history=model.fit(x_train, y_train2,
          epochs=400,
          validation_split=0.3,  # 如是 0.3，在訓練時會拿 30% 的數據自行驗證數據)
          batch_size=10)

#測試
score = model.evaluate(x_test, y_test2)
print("score:",score)

predict = model.predict(x_test)
print("Ans:",np.argmax(predict,axis=-1))

"""
predict2 = model.predict_classes(x_test)
print("predict_classes:",predict2)
print("y_test",y_test[:])
"""

import matplotlib.pyplot as plt
import myfun
myfun.pyplot_中文()
# loss是訓練集的損失值，val_loss是測試集的損失值
plt.plot(history.history['accuracy'],label="訓練時的正確率(accuracy)")
plt.plot(history.history['loss'],label="訓練時的損失率(loss)")
plt.plot(history.history['val_accuracy'],label="測試時的驗證正確率(val_accuracy)")
plt.plot(history.history['val_loss'],label="測試時的驗證損失率(val_loss)")

plt.title('model accuracy')
plt.ylabel('accuracy , loss, validation accuracy, validation loss')
plt.xlabel('epoch')
plt.legend(loc='lower left')
plt.show()
