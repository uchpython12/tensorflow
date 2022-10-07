import tensorflow as tf
import numpy as np
def CreateDatasets(high,iNum,iArraySize):
    x_train = np.random.random((iNum, iArraySize)) * float(high)
    y_train = ((x_train[:iNum,0]+x_train[:iNum,1])/2).astype(int)     # 取整數
    return x_train, y_train,tf.keras.utils.to_categorical(y_train, num_classes=(high))
category=10
dim=2
x_train,y_train,y_train2=CreateDatasets(category,1000*10,dim) # 修改這裡


# 建立模型
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(units=10,
    activation=tf.nn.relu,
    input_dim=dim))
model.add(tf.keras.layers.Dense(units=10*100,
    activation=tf.nn.relu ))
model.add(tf.keras.layers.Dense(units=category,
    activation=tf.nn.softmax ))
model.compile(optimizer='adam',
    loss=tf.keras.losses.categorical_crossentropy,
    metrics=['accuracy'])
model.fit(x_train, y_train2,
          epochs=20*10,
          batch_size=64)

#測試
x_test,y_test,y_test2=CreateDatasets(category,10,dim)
score = model.evaluate(x_test, y_test2, batch_size=128)
print("score:",score)

predict = model.predict(x_test)
#print("predict:",predict)
print("Ans:",np.argmax(predict[0]),np.argmax(predict[1]),np.argmax(predict[2]),np.argmax(predict[3]))

predict2 = model.predict_classes(x_test)
print("predict_classes:",predict2)
print("y_test",y_test[:])




