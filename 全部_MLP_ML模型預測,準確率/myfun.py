import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
from tensorflow.keras.callbacks import TensorBoard
import matplotlib.pyplot as plt # 匯入matplotlib 的pyplot 類別，並設定為plt
from sklearn.cluster import KMeans
from sklearn import metrics
from tensorflow.keras.models import model_from_json
import pickle

def pyplot_中文():
    import sys
    if sys.platform.startswith("linux"):
        print("linux")
    elif sys.platform == "darwin":
        # MAC OS X
        try:
            import seaborn as sns
            sns.set(font="Arial Unicode MS")  # "DFKai-SB"
            print("Initiated Seaborn font")
        except:
            print("Initiated Seaborn font failed")
        try:
            import matplotlib.pyplot as plt
            from matplotlib.font_manager import FontProperties
            plt.rcParams['font.sans-serif'] = 'Arial Unicode MS'
            plt.rcParams['axes.unicode_minus'] = False
            print("Initiated matplotlib font")
        except:
            print("Initiated matplotlib font failed")

    elif sys.platform == "win32":
        # Windows (either 32-bit or 64-bit)
        try:
            import seaborn as sns
            sns.set(font="sans-serif")  # "DFKai-SB"
            print("Initiated Seaborn font ")
        except:
            print("Initiated Seaborn font failed")
        try:
            import matplotlib.pyplot as plt
            plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']  # 換成中文的字體
            plt.rcParams['axes.unicode_minus'] = False  # 步驟二（解決seaborn座標軸亂碼問題）
            print("Initiated matplotlib font")
        except:
            print("Initiated matplotlib font failed")

def ML_read_excel(fileName, featuresCol, labelCol):
    df = pd.read_excel(fileName)  # 讀取 pandas資料
    print(df.columns)  # 印出所有列
    x = df[featuresCol]  # 設定x的資料
    print(x)  # 印出x的資料
    y = df[labelCol]  # 設定y的資料
    print(y)
    ###   Pandas 轉 numpy
    x = x.to_numpy()  # x從 pandas 轉 numpy
    print(x)  # 印出 轉 numpy後結果

    y = y.to_numpy()  # y從 pandas 轉 numpy
    print(y)
    print("資料筆數為", y.shape)
    y = y.reshape(y.shape[0])  # 將 y=y.to_numpy() 二維陣列,改為一維陣列
    print(y, "外型大小")  # 印出結果
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.05)
    return train_x, test_x, train_y, test_y
    # #印出y的資料

def AI_MLP_Classification(train_x, test_x, train_y, test_y):
    dim=train_x.shape[1]  # 4個欄位 特徵值
    category = len(np.unique(train_y))  #3個答案

    train_y2 = tf.keras.utils.to_categorical(train_y, num_classes=(category))
    test_y2 = tf.keras.utils.to_categorical(test_y, num_classes=(category))

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
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss=tf.keras.losses.categorical_crossentropy,
                  metrics=['accuracy'])
    tensorboard = TensorBoard(log_dir="logs")
    history = model.fit(train_x, train_y2,
                        epochs=200, batch_size=128,
                        callbacks=[tensorboard],
                        verbose=1)

    # 保存模型架構
    with open("MLP_model.json", "w") as json_file:
        json_file.write(model.to_json())
    # 保存模型權重
    model.save_weights("MLP_model.h5")
    predict = model.predict(test_x)
    print("預測:", np.argmax(predict, axis=-1))
    print("實際", test_y)
    print("準確率:", model.evaluate(test_x, test_y2, batch_size=128))
    return model

    # 決策樹
def ML_Classification_DecisionTreeClassifier(train_x, test_x, train_y, test_y):

    from sklearn import tree
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(train_x, train_y.ravel())
    prediction = clf.predict(test_x)
    clfScore = clf.score(test_x, test_y)
    print("決策樹 預估答案       ：", prediction, " 準確率：", clfScore)

    #####
    tree.export_graphviz(clf, out_file='決策樹.dot')
    # 換成中文的字體
    pyplot_中文()
    fig = plt.figure()
    _ = tree.plot_tree(clf,
                       # feature_names=colX,
                       # class_names=colY2,
                       filled=True)
    fig.savefig("決策樹1.png")
    plt.show()

    # KMeans 演算法
    kmeans = KMeans(n_clusters=3)
    kmeans.fit(train_x)
    y_predict = kmeans.predict(test_x)
    kmeansScore = metrics.accuracy_score(test_y, kmeans.predict(test_x))
    kmeanshomogeneity_score = metrics.homogeneity_score(test_y, kmeans.predict(test_x))  # 修正答案
    print("KMeans 演算法 預估答案：", y_predict, " 準確率：", kmeansScore)
    print("KMeans 演算法 預估答案：", y_predict, " 修正後準確率：", kmeanshomogeneity_score)
    return clf

def ML_Classification_DecisionTreeClassifier_entropy(train_x, test_x, train_y, test_y):
    from sklearn import tree
    # 決策樹 演算法
    clf = tree.DecisionTreeClassifier(criterion='entropy', splitter='random', max_depth=2)
    clf = clf.fit(train_x, train_y)
    tree.export_graphviz(clf, out_file='tree-C2.dot')
    clfPredict = clf.predict(test_x)
    clfScore2 = clf.score(test_x, test_y)
    print("決策樹 2 演算法 預估答案：", clfPredict, " 準確率：", clfScore2)

    #######
    fig = plt.figure()
    _ = tree.plot_tree(clf,
                       # feature_names=colX,
                       # class_names=colY,
                       filled=True)

    fig.savefig("decistion_tree.png")
    plt.show()
    return clf
"""
p 內定值=2
Euclidean Distance
距離＝  √((x1-x2)**2+(y1-y2)**2)


p =1
Manhattan Distance
距離＝｜x1-x2｜+|y1-y2|
"""


def ML_Classification_DecisionTreeClassifier_gini(train_x, test_x, train_y, test_y):
    from sklearn import tree
    # 決策樹 演算法

    # https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier
    clf = tree.DecisionTreeClassifier(criterion='gini')
    clf = clf.fit(train_x, train_y)
    tree.export_graphviz(clf, out_file='tree-C1.dot')
    clfPredict = clf.predict(test_x)
    clfScore1 = clf.score(test_x, test_y)
    print("決策樹 1 演算法 預估答案：", clfPredict, " 準確率：", clfScore1)
    return clf
def ML_Classification_KNeighborsClassifier(train_x, test_x, train_y, test_y):
    # KNN 演算法

    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors=3, p=1)
    knn.fit(train_x, train_y)
    knnPredict = knn.predict(test_x)
    knnScore = knn.score(test_x, test_y)
    print("KNN    演算法 預估答案：", knnPredict, " 準確率：", knnScore)
    return knn
def ML_Classification_RandomForestClassifier(train_x, test_x, train_y, test_y):
    # 隨機森林

    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(n_estimators=100, max_depth=10,
                                random_state=2)
    rf.fit(train_x, train_y.ravel())
    prediction = rf.predict(test_x)
    rfScore = rf.score(test_x, test_y)
    print("隨機森林 預估答案      ：", prediction, " 準確率：", rfScore)
    from sklearn.tree import export_graphviz
    export_graphviz(rf.estimators_[2], out_file='隨機森林1.dot',
                    # feature_names=colX,
                    # class_names =colY2,
                    rounded=True, proportion=False,
                    precision=2, filled=True)

    from sklearn import tree
    fig, axes = plt.subplots(nrows=1, ncols=5)
    for index in range(0, 5):
        tree.plot_tree(rf.estimators_[index],
                       # feature_names=colX,
                       # class_names=colY2,
                       filled=True,
                       ax=axes[index])

        axes[index].set_title('Estimator: ' + str(index), fontsize=11)
    fig.savefig('隨機森林1.png')
    plt.show()
    return rf
def AI_MLP_Classification_load(test_x,test_y,modelFile="MLP_model.json",weightsFile="MLP_model.h5"):
    test_y = tf.keras.utils.to_categorical(test_y, num_classes=(len(np.unique(test_y))))
    # 讀取模型架構
    json_file = open(modelFile, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # 讀取模型權重
    model.load_weights(weightsFile)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss=tf.keras.losses.categorical_crossentropy,
                  metrics=['accuracy'])

    predict = model.predict(test_x)
    print("MLP 預測答案:", np.argmax(predict, axis=-1))
    print("準確率:", model.evaluate(test_x, test_y, batch_size=128))
    return  model


def ML_Classification_load(test_x,test_y,modelFile):
    model = pickle.load(open(modelFile, 'rb'))
    y_predict = model.predict(test_x)
    score = model.score(test_x,test_y)
    print(modelFile,"預測結果:",y_predict,"準確率",score)
    return model
