import numpy as np
import myfun


category=3
dim=4
test_x = np.array([[5.1,3.5,1.4,0.2],[7.0,3.2,4.7,1.4],[6.9,3.1,5.4,2.1],[5.9,3.0,5.1,1.8]])
test_y =  np.array([0,1,2,2])

model= myfun.AI_MLP_Classification_load(test_x, test_y,modelFile="MLP_model.json", weightsFile="MLP_model.h5")  # 讀取MLP模型架構
model= myfun.ML_Classification_load(test_x, test_y,modelFile="DecisionTreeClassifier.model")# 讀取決策樹模型架構
model= myfun.ML_Classification_load(test_x, test_y,modelFile="DecisionTreeClassifier_entropy.model")# 讀取決策樹模型架構
model= myfun.ML_Classification_load(test_x, test_y,modelFile="DecisionTreeClassifier_gini.model")# 讀取決策樹模型架構
model= myfun.ML_Classification_load(test_x,test_y, modelFile="KNeighborsClassifier.model")# 讀取KNN模型架構
model= myfun.ML_Classification_load(test_x, test_y,modelFile="RandomForestClassifier.model")# 讀取隨機森林模型架構







