from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def ML_read_dataframe_標準化(fileName, featuresCol, labelCol):
    x, y = ML_read_excel_No_split(fileName, featuresCol, labelCol)
    x, scaler = ML_標準化1_轉換(x)
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.05)
    return train_x, test_x, train_y, test_y,scaler

# 讀取資料，有沒拆分
def ML_read_excel_No_split(fileName, featuresCol, labelCol):
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
    return x, y

def ML_標準化1_轉換(x):
    scaler = MinMaxScaler()  # 初始化
    scaler.fit(x)  # 找標準化範圍
    x1 = scaler.transform(x)  # 把資料轉換
    return x1, scaler