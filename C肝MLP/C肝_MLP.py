import tensorflow as tf
import numpy as np
import ML_myfun


print("====下載資料==============")
col = ['Age','Sex','ALB','ALP','ALT','AST','BIL','CHE','CHOL','CREA','GGT','PROT']
col_target=['Category']

print("====讀取資料==標準化============")

train_x, test_x, train_y, test_y,scaler=ML_myfun.ML_read_dataframe_標準化("C肝.xlsx", col, col_target)
print("外型大小",train_x.shape,test_x.shape,train_y.shape,test_y.shape)
print("前面幾筆:",train_x)

category=len(np.unique(train_y)) # train_y_不同元素數量 5種 答案
dim=len(col) #特徵值(欄位) 12個

# 將數字轉為 One-hot 向量
train_y2=tf.keras.utils.to_categorical(train_y, num_classes=(category))
test_y2=tf.keras.utils.to_categorical(test_y, num_classes=(category))

print("train_x[:4]",train_x[:12])
print("train_y[:4]",train_y[:12])
print("train_y2[:4]",train_y2[:12])


# 建立模型
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(units=1000,
    activation=tf.nn.relu,
    input_dim=dim))
model.add(tf.keras.layers.Dense(units=1000,
    activation=tf.nn.relu ))
model.add(tf.keras.layers.Dense(units=1000,
    activation=tf.nn.relu ))
model.add(tf.keras.layers.Dense(units=category,
    activation=tf.nn.softmax ))
model.compile(optimizer='adam',
    loss=tf.keras.losses.categorical_crossentropy,
    metrics=['accuracy'])
model.fit(train_x, train_y2,
          epochs=300,
          batch_size=100)

#測試
score = model.evaluate(test_x, test_y2, batch_size=128)
print("score:",score)

predict = model.predict(test_x)
print("Ans:",np.argmax(predict,axis=-1))

print("實際",test_y)

