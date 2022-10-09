import myfun
import pickle

train_x, test_x, train_y, test_y=\
              myfun.ML_read_excel("Iris.xlsx",
              ["SepalLengthCm","SepalWidthCm","PetalLengthCm","PetalWidthCm"],
              ["Species"])

model=myfun.AI_MLP_Classification(train_x, test_x, train_y, test_y)# MLP
DecisionTreeClassifier=myfun.ML_Classification_DecisionTreeClassifier(train_x, test_x, train_y, test_y)# 決策樹
DecisionTreeClassifier_entropy=myfun.ML_Classification_DecisionTreeClassifier_entropy(train_x, test_x, train_y, test_y)# 決策樹 演算法
DecisionTreeClassifier_gini=myfun.ML_Classification_DecisionTreeClassifier_gini(train_x, test_x, train_y, test_y)# 決策樹 演算法
KNeighborsClassifier=myfun.ML_Classification_KNeighborsClassifier(train_x, test_x, train_y, test_y)# KNN 演算法
RandomForestClassifier=myfun.ML_Classification_RandomForestClassifier(train_x, test_x, train_y, test_y)# 隨機森林

#  save 演算法 和權重
pickle.dump(DecisionTreeClassifier, open("DecisionTreeClassifier.model", 'wb'))
pickle.dump(DecisionTreeClassifier_entropy, open("DecisionTreeClassifier_entropy.model", 'wb'))
pickle.dump(DecisionTreeClassifier_gini, open("DecisionTreeClassifier_gini.model", 'wb'))
pickle.dump(KNeighborsClassifier, open("KNeighborsClassifier.model", 'wb'))
pickle.dump(RandomForestClassifier, open("RandomForestClassifier.model", 'wb'))

"""
# 讀取 機器學習演算法 和 權重

import pickle
loaded_model = pickle.load(open("knn.model", 'rb'))
"""
