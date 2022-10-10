import io
import matplotlib.pyplot as plt  # 匯入matplotlib 的pyplot 類別，並設定為plt
import xml.etree.ElementTree as ET
from matplotlib.patches import Shadow
from matplotlib.patches import ConnectionPatch
import tkinter as tk
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # GUI 元件
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn import metrics
# Import seaborn
import seaborn as sns
import matplotlib.pyplot as plt # 匯入matplotlib 的pyplot 類別，並設定為plt
import numpy as np
# Apply the default theme
sns.set_theme()



def hat_graph(ax, xlabels, values, group_labels):
    """
    Create a hat graph.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The Axes to plot into.
    xlabels : list of str
        The category names to be displayed on the x-axis.
    values : (M, N) array-like
        The data values.
        Rows are the groups (len(group_labels) == M).
        Columns are the categories (len(xlabels) == N).
    group_labels : list of str
        The group labels displayed in the legend.
    """

    def label_bars(heights, rects):
        """Attach a text label on top of each bar."""
        for height, rect in zip(heights, rects):
            ax.annotate(f'{height}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 4),  # 4 points vertical offset.
                        textcoords='offset points',
                        ha='center', va='bottom')

    values = np.asarray(values)
    x = np.arange(values.shape[1])
    ax.set_xticks(x, labels=xlabels)
    spacing = 0.3  # spacing between hat groups
    width = (1 - spacing) / values.shape[0]
    heights0 = values[0]
    for i, (heights, group_label) in enumerate(zip(values, group_labels)):
        style = {'fill': False} if i == 0 else {'edgecolor': 'black'}
        rects = ax.bar(x - spacing / 2 + i * width, heights - heights0,
                       width, bottom=heights0, label=group_label, **style)
        label_bars(heights, rects)


# 圖一
def char1(listDate1, listy, label1, label2, label3, title, ylabel, xlabel):
    plt.plot(listDate1, listy[0], "y--", label=label1)  # 建立圖表 x軸listDate1 y軸listy[0]
    plt.plot(listDate1, listy[1], "rp--", label=label2)  # 建立圖表 x軸 listDate1y軸listy[1]
    plt.plot(listDate1, listy[2], "cd--", label=label3)  # 建立圖表 x軸 listDate1y軸listy[2]
    plt.legend(loc='upper right')  # 在右上角顯示標籤
    plt.xlabel('七月確診日期')
    plt.ylabel('單日確診人數')

    plt.title('新冠肺炎19 台北,桃園,新竹 確診數量圖表 ')
    plt.savefig("新冠肺炎確診數量圖.jpg")


# 圖二
def char2(listDate1, listy1, listy2, listy3, title, ylabel, xlabel, label1, label2, label3):
    x = listDate1
    y1 = listy1
    max = 3
    plt.bar(x, y1,
            alpha=0.5,
            width=1 / max, edgecolor="black",
            linewidth=0.7, label=label1
            )
    x2 = [i + (1 / max) for i in x]
    y2 = listy2
    plt.bar(x2, y2,
            alpha=0.5,
            width=1 / max, edgecolor="white",
            linewidth=0.7, label=label2)
    x3 = [i + (2 / max) for i in x]
    y3 = listy3
    plt.bar(x3, y3,
            alpha=0.5,
            width=1 / max, edgecolor="red",
            linewidth=0.7, label=label3)
    plt.legend(loc='upper right')  # 在右上角顯示標籤
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)


# 圖九
def char3(listy, listDate1, label1, label2, label3, xlabel, ylabel, title):
    plt.plot(listDate1, listy[0], "go", label=label1)
    plt.plot(listDate1, listy[1], "r_", label=label2)
    plt.plot(listDate1, listy[2], "y-", label=label3)
    plt.legend()  # 自動改變顯示的位置

    plt.title(title)
    plt.ylabel(xlabel)  # 顯示Y 座標的文字
    plt.xlabel(ylabel)  # 顯示Y 座標的文字


def char4(x, y, label1, label2, label3, xlabel, ylabel, title):
    # def charts(x,y,title,xlabel,ylabel):
    # import matplotlib.pyplot as plt
    ### 第一張圖
    # plt.subplot(2, 2, 1, facecolor='y')
    list1 = ['ro', "go", "bo", "r-", "g-", "b-"]
    i = 0
    for y2 in y:
        plt.plot(x, y2, list1[i])
        i = i + 1

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(ylabel)
    plt.xticks(rotation=-90, fontsize=8)


def char5(x, y, label1, label2, label3, xlabel, ylabel, title):
    ### 第2張圖
    # plt.subplot(2, 2, 2, facecolor='k')

    plt.bar(x, y[0], width=1,
            color="blue",  # 顏色 #ff0000 rgb 三原色
            alpha=0.9,  # 透明度
            edgecolor="white", linewidth=0.7)
    plt.xlabel(title[0])
    plt.ylabel(xlabel[0])
    plt.title(ylabel[0])
    plt.xticks(rotation=-90, fontsize=8)


def char6(x, y, label1, label2, label3, xlabel, ylabel, title):
    #### 第3張圖
    # plt.subplot(2, 2, 3)
    plt.plot(x, y[0], 'b|')
    plt.xlabel(title[0])
    plt.ylabel(xlabel[0])
    plt.title(ylabel[0])
    plt.xticks(rotation=-90, fontsize=8)


def char7(x, y, label1, label2, label3, xlabel, ylabel, title):
    ### 第4張圖
    # plt.subplot(2, 2, 4)
    plt.pie(y[0], labels=x,
            radius=1,  # 半徑
            center=(4, 4),  # 中心點
            wedgeprops={"linewidth": 1,
                        "edgecolor": "white"},
            frame=True)

    plt.xlabel(title[0])
    plt.ylabel(xlabel[0])
    plt.title(ylabel[0])
    plt.xticks(rotation=-90, fontsize=8)


def char8(x, y, label1, label2, label3, xlabel, ylabel, title):
    # def charts(x,y,title,xlabel,ylabel):
    # import matplotlib.pyplot as plt
    ### 第一張圖
    # plt.subplot(2, 2, 1, facecolor='y')

    plt.plot(x, y[0], "r--")

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(ylabel)
    plt.xticks(rotation=-90, fontsize=8)


def char9(x, y, label1, label2, label3, xlabel, ylabel, title):
    # def charts(x,y,title,xlabel,ylabel):
    # import matplotlib.pyplot as plt
    ### 第一張圖
    # plt.subplot(2, 2, 1, facecolor='y')

    plt.plot(x, y[0], "r*")

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(ylabel)
    plt.xticks(rotation=-90, fontsize=8)


"""
charts(list民宿店名,list低價位,
       ['烏來區民宿','烏來區民宿','烏來區民宿','烏來區民宿'],
       ['民宿店名','民宿店名','民宿店名','民宿店名'],
       ['低價位','低價位','低價位','低價位'])
"""


# 圖9
def subplots_char1(listy1, listDate1, label1, xlabel, ylabel, title, ax=None):
    if ax is None:
        fig, ax = plt.subplots()  # 建立圖表 畫面分割成1個
    ax.plot(listDate1, listy1, "ro")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)


# 圖9
def subplots_char2(listy1, listDate1, label1, xlabel, ylabel, title, ax=None):
    if ax is None:
        fig, ax = plt.subplots()  # 建立圖表 畫面分割成1個
    ax.plot(listDate1, listy1, "b--")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)


# 圖三
def subplots_char3(listDate1, listy1, listy2, listy3, title, ylabel, xlabel, label1, label2, label3, ax=None):
    xlabels = listDate1
    Taipei = np.array(listy1)
    Taoyun = np.array(listy2)
    Hsinchu = np.array(listy3)
    if ax is None:
        fig, ax = plt.subplots()
    hat_graph(ax, xlabels, [Taipei, Taoyun, Hsinchu], [label1, label2, label3])

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 3000)
    ax.set_title(title)
    ax.legend()


# 圖四
def subplots_char4(listDate1, listy, title, list1Label, ax=None):
    vegetables = list1Label
    farmers = listDate1

    harvest = np.array([listy[0], listy[1], listy[2]])

    ax輸入為空 = None
    if ax is None:
        fig, ax = plt.subplots()
        ax輸入為空 = True
    else:
        ax輸入為空 = False
    im = ax.imshow(harvest)

    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(len(farmers)), labels=farmers)
    ax.set_yticks(np.arange(len(vegetables)), labels=vegetables)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(vegetables)):
        for j in range(len(farmers)):
            text = ax.text(j, i, harvest[i, j],
                           ha="center", va="center", color="r")

    ax.set_title(title)
    if ax輸入為空 == True:
        fig.tight_layout()


# 圖五
def subplots_char5(list1Label, listy1, listy2, listy3, ax=None):
    if ax is not None:
        ax.plot(listy1)
    else:
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])

        labels = list1Label
        fracs = [listy1[0], listy2[0], listy3[0]]

        explode = (0, 0.05, 0)

        # We want to draw the shadow for each pie but we will not use "shadow"
        # option as it doesn't save the references to the shadow patches.
        pies = ax.pie(fracs, explode=explode, labels=labels, autopct='%1.1f%%')

        for w in pies[0]:
            # set the id with the label.
            w.set_gid(w.get_label())

            # we don't want to draw the edge of the pie
            w.set_edgecolor("none")

        for w in pies[0]:
            # create shadow patch
            s = Shadow(w, -0.01, -0.01)
            s.set_gid(w.get_gid() + "_shadow")
            s.set_zorder(w.get_zorder() - 0.1)
            ax.add_patch(s)

        # save
        f = io.BytesIO()
        plt.savefig(f, format="svg")

        # Filter definition for shadow using a gaussian blur and lighting effect.
        # The lighting filter is copied from http://www.w3.org/TR/SVG/filters.html

        # I tested it with Inkscape and Firefox3. "Gaussian blur" is supported
        # in both, but the lighting effect only in Inkscape. Also note
        # that, Inkscape's exporting also may not support it.

        filter_def = """
          <defs xmlns='http://www.w3.org/2000/svg'
                xmlns:xlink='http://www.w3.org/1999/xlink'>
            <filter id='dropshadow' height='1.2' width='1.2'>
              <feGaussianBlur result='blur' stdDeviation='2'/>
            </filter>

            <filter id='MyFilter' filterUnits='objectBoundingBox'
                    x='0' y='0' width='1' height='1'>
              <feGaussianBlur in='SourceAlpha' stdDeviation='4%' result='blur'/>
              <feOffset in='blur' dx='4%' dy='4%' result='offsetBlur'/>
              <feSpecularLighting in='blur' surfaceScale='5' specularConstant='.75'
                   specularExponent='20' lighting-color='#bbbbbb' result='specOut'>
                <fePointLight x='-5000%' y='-10000%' z='20000%'/>
              </feSpecularLighting>
              <feComposite in='specOut' in2='SourceAlpha'
                           operator='in' result='specOut'/>
              <feComposite in='SourceGraphic' in2='specOut' operator='arithmetic'
            k1='0' k2='1' k3='1' k4='0'/>
            </filter>
          </defs>
        """

        tree, xmlid = ET.XMLID(f.getvalue())

        # insert the filter definition in the svg dom tree.
        tree.insert(0, ET.XML(filter_def))

        for i, pie_name in enumerate(labels):
            pie = xmlid[pie_name]
            pie.set("filter", 'url(#MyFilter)')

            shadow = xmlid[pie_name + "_shadow"]
            shadow.set("filter", 'url(#dropshadow)')

        fn = "svg_filter_pie.svg"
        print(f"Saving '{fn}'")
        ET.ElementTree(tree).write(fn)


# 圖六
def subplots_char6(list1Label, listy1, listy2, listy3, ax=None):
    # Some data
    labels = list1Label
    fracs = [listy1[0], listy2[0], listy3[0]]

    # Make figure and axes
    if ax is None:
        fig, axs = plt.subplots(2, 2)
        ax1 = axs[0, 0]
        ax2 = axs[0, 1]
        ax3 = axs[1, 0]
        ax4 = axs[1, 1]
    else:
        ax1 = ax
        ax2 = None
        ax3 = None
        ax4 = None

    # A standard pie plot
    ax1.pie(fracs, labels=labels, autopct='%1.1f%%', shadow=True)

    # Shift the second slice using explode
    if ax2 is not None:
        ax2.pie(fracs, labels=labels, autopct='%.0f%%', shadow=True,
                explode=(0, 0.1, 0))

    # Adapt radius and text size for a smaller pie
    if ax3 is not None:
        patches, texts, autotexts = ax3.pie(fracs, labels=labels,
                                            autopct='%.0f%%',
                                            textprops={'size': 'smaller'},
                                            shadow=True, radius=0.5)
        # Make percent texts even smaller
        plt.setp(autotexts, size='x-small')
        autotexts[0].set_color('white')

    # Use a smaller explode and turn of the shadow for better visibility
    if ax4 is not None:
        patches, texts, autotexts = ax4.pie(fracs, labels=labels,
                                            autopct='%.0f%%',
                                            textprops={'size': 'smaller'},
                                            shadow=False, radius=0.5,
                                            explode=(0, 0.05, 0))
        plt.setp(autotexts, size='x-small')
        autotexts[0].set_color('white')


# 圖七
def subplots_char7(list1Label, listy1, listy2, listy3, ax=None):
    # make figure and assign axis objects
    if ax is None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 5))
        fig.subplots_adjust(wspace=0)
    else:
        ax1 = ax
        ax2 = None

    # pie chart parameters
    overall_ratios = [listy1[0], listy2[0], listy3[0]]
    labels = list1Label
    explode = [0.1, 0, 0]
    # rotate so that first wedge is split by the x-axis
    angle = -180 * overall_ratios[0]
    wedges, *_ = ax1.pie(overall_ratios, autopct='%1.1f%%', startangle=angle,
                         labels=labels, explode=explode)

    # bar chart parameters
    age_ratios = [.33, .54, .07, .06]
    age_labels = ['Under 35', '35-49', '50-65', 'Over 65']
    bottom = 1
    width = .2

    # Adding from the top matches the legend.
    if ax2 is not None:
        for j, (height, label) in enumerate(reversed([*zip(age_ratios, age_labels)])):
            bottom -= height
            bc = ax2.bar(0, height, width, bottom=bottom, color='C0', label=label,
                         alpha=0.1 + 0.25 * j)
            ax2.bar_label(bc, labels=[f"{height:.0%}"], label_type='center')

        ax2.set_title('Age of approvers')
        ax2.legend()
        ax2.axis('off')
        ax2.set_xlim(- 2.5 * width, 2.5 * width)

        # use ConnectionPatch to draw lines between the two plots
        theta1, theta2 = wedges[0].theta1, wedges[0].theta2
        center, r = wedges[0].center, wedges[0].r
        bar_height = sum(age_ratios)

        # draw top connecting line
        x = r * np.cos(np.pi / 180 * theta2) + center[0]
        y = r * np.sin(np.pi / 180 * theta2) + center[1]
        con = ConnectionPatch(xyA=(-width / 2, bar_height), coordsA=ax2.transData,
                              xyB=(x, y), coordsB=ax1.transData)
        con.set_color([0, 0, 0])
        con.set_linewidth(4)
        ax2.add_artist(con)

        # draw bottom connecting line
        x = r * np.cos(np.pi / 180 * theta1) + center[0]
        y = r * np.sin(np.pi / 180 * theta1) + center[1]
        con = ConnectionPatch(xyA=(-width / 2, 0), coordsA=ax2.transData,
                              xyB=(x, y), coordsB=ax1.transData)
        con.set_color([0, 0, 0])
        ax2.add_artist(con)
        con.set_linewidth(4)


# 圖八
def subplots_char8(listy1, listDate1, label1, xlabel, ylabel, title, ax=None):
    if ax is None:
        fig, ax = plt.subplots()  # 建立圖表 畫面分割成1個

    x = listDate1
    y1 = listy1
    ax.bar(x, y1,
           alpha=0.5,
           width=1, edgecolor="black",
           linewidth=0.7, label=label1
           )

    ax.legend(loc='upper right')  # 在右上角顯示標籤

    # ax.xlabel(xlabel)
    # ax.ylabel(ylabel)

    # ax.title(title)


# 圖9
def subplots_char9(listy1, listDate1, label1, xlabel, ylabel, title, ax=None):
    if ax is None:
        fig, ax = plt.subplots()  # 建立圖表 畫面分割成1個
    ax.plot(listDate1, listy1)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)


# 九宮格
def NineCharts(list1Label, listDate1, listy, listy1, listy2, listy3, title, ylabel, xlabel, label1, label2, label3):
    plt.subplot(3, 3, 1)  # , facecolor='y')
    char1(listDate1, listy, label1, label2, label3, title, ylabel, xlabel)
    plt.subplot(3, 3, 2)
    char2(listDate1, listy1, listy2, listy3, title, ylabel, xlabel, label1, label2, label3)
    plt.subplot(3, 3, 3)
    char3(listy, listDate1, label1, label2, label3, xlabel, ylabel, title)
    plt.subplot(3, 3, 4)
    char4(listDate1, listy, label1, label2, label3, xlabel, ylabel, title)
    plt.subplot(3, 3, 5)
    char5(listDate1, listy, label1, label2, label3, xlabel, ylabel, title)
    plt.subplot(3, 3, 6)
    char6(listDate1, listy, label1, label2, label3, xlabel, ylabel, title)
    plt.subplot(3, 3, 7)
    char7(listDate1, listy, label1, label2, label3, xlabel, ylabel, title)
    plt.subplot(3, 3, 8)
    char8(listDate1, listy, label1, label2, label3, xlabel, ylabel, title)
    plt.subplot(3, 3, 9)
    char9(listDate1, listy, label1, label2, label3, xlabel, ylabel, title)

    """
    plt.subplot(3, 3, 4)
    char4(listDate1, listy, title, list1Label)

    plt.subplot(3, 3, 5)
    char5(list1Label, listy1, listy2, listy3)
    plt.subplot(3, 3, 6)
    char6(list1Label, listy1, listy2, listy3)
    plt.subplot(3, 3, 7)
    char7(list1Label, listy1, listy2, listy3)
    plt.subplot(3, 3, 8)
    char8(listy1, listDate1, label1, xlabel, ylabel, title)
    plt.subplot(3, 3, 9)
    char9(listy, listDate1, label1, label2, label3, xlabel, ylabel, title)
    """


# 九宮格
def subplots_NineCharts(list1Label, listDate1, listy, listy1, listy2, listy3, title, ylabel, xlabel, label1, label2,
                        label3):
    fig, ax = plt.subplots(3, 3)  # 建立圖表 畫面分割成1個
    subplots_char1(listy1, listDate1, label1, xlabel, ylabel, title, ax[0, 0])
    subplots_char2(listy1, listDate1, label1, xlabel, ylabel, title, ax[0, 1])
    subplots_char3(listDate1, listy1, listy2, listy3, title, ylabel, xlabel, label1, label2, label3, ax[0, 2])
    subplots_char4(listDate1, listy, title, list1Label, ax[1, 0])
    subplots_char5(list1Label, listy1, listy2, listy3, ax[1, 1])
    subplots_char6(list1Label, listy1, listy2, listy3, ax[1, 2])
    subplots_char7(list1Label, listy1, listy2, listy3, ax[2, 0])
    subplots_char8(listy1, listDate1, label1, xlabel, ylabel, title, ax[2, 1])
    subplots_char9(listy1, listDate1, label1, xlabel, ylabel, title, ax[2, 2])


def pandas_取得裡面的種類(df, columeName):
    list1 = df[columeName].unique()
    return list1


def pandas_文字轉數字(df, columeName):
    dfvalue = df[columeName].rank(method='dense', ascending=False).astype(int)
    return dfvalue

def Array_ToDic(lst):
    res_dct = {lst[i]: i for i in range(0, len(lst), 1)}
    return res_dct

def pandas_文字轉數字B(df,colName):
    return  df[colName].map(Array_ToDic(df[colName].unique()))



def pandas_where_demo(df):
    t1 = np.where((df[:, 1] == "2022-07-23") & (df[:, 3] == "台中市"))
    x = df[t1]
    return x


def pandas_Col_Rename_替換字串(df, replace1, replace2):
    dict1 = {}
    columns = df.columns
    for column in columns:
        t1 = column.replace(replace1, replace2)
        dict1[column] = t1
    df.rename(columns=dict1, inplace=True)
    return df


"""
print("移除掉空的資料前", df1.shape)
    age_no_na = df1[df1.notna()]
    print("移除掉空的資料後", age_no_na.shape)
    print("移除掉空的資料", age_no_na)

    age_na1 = df1.isnull()
    print("空的資料", age_na1)

    age_na2 = df1.isnull().values.any()
    print("空的資料", age_na2)

    print(df1)

    print("-----方法2 補上0------")
    df2 = df1.fillna(0)  # 如果是空的 補上0
    print(df2)

    print("-----方法2  移除 row------")
    df3 = df1.dropna()
    print(df3)
    print("-----方法2  移除 col------")
    df4 = df1.dropna(axis=1)
    print(df4)


"""


def pandas_nan_col_是否有空的(df, columeName):
    df = df[df[columeName].isna()]
    return df


def pandas_nan_col_移除(df, columeName):
    df = df[df[columeName].isna()]
    return df


def pandas_nan_移除該筆資料(df1):
    df3 = df1.dropna()
    return df3


def pandas_nan_col_移除該筆資料(df1, columeName):
    df3 = df1.dropna(subset=[columeName])
    return df3


def pandas_nan_col_替換(df, columeName, value1):
    df[columeName] = df[columeName].replace(np.nan, value1)
    return df


def pandas_nan_填值(df, value1):
    df = df.fillna(value1)
    # df[columeName] = df[columeName].replace(np.nan, value1)
    return df


def pandas_替換(df, replace1, replace2):
    df = df.replace(replace1, replace2)
    return df

def pandas_替換_col(df, col,replace1, replace2):
    # df['Sex'] = df['Sex'].replace('m', 1)
    t1 = df[col].replace(replace1, replace2)
    return t1



def pandas_To_numpy(df):
    a = df.to_numpy()
    return a


def pandas_File_read_xlsx(fileName, tableName):
    df1 = pd.read_excel(fileName, tableName, header=0)
    # print(df1.head(5))
    return df1


def pandas_File_write_xlsx(df, fileName, tableName):
    from pandas import ExcelWriter
    writer = ExcelWriter(fileName, engine='xlsxwriter')
    df.to_excel(writer, sheet_name=tableName, index=False)
    writer.save()


def ML_read_dataframe(df, featuresCol, labelCol):
    # print(df.columns)  # 印出所有列
    # print(df[["sepal length (cm)","sepal width (cm)","petal length (cm)","petal width (cm)"]])     #將所有列的資料印出
    x = df[featuresCol]  # 設定x的資料
    print(x)  # 印出x的資料
    y = df[labelCol]  # 設定y的資料
    print(y)
    ###   Pandas 轉 numpy
    x = x.to_numpy()  # x從 pandas 轉 numpy (參考Day34-524)
    print(x)  # 印出 轉 numpy後結果

    y = y.to_numpy()  # y從 pandas 轉 numpy
    print(y)
    print("資料筆數為", y.shape)
    y = y.reshape(y.shape[0])  # 將 y=y.to_numpy() 二維陣列,改為一維陣列(參考Day37-573)
    print(y, "外型大小")  # 印出結果
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.05)
    return train_x, test_x, train_y, test_y

def ML_read_dataframe_標準化xy(fileName, featuresCol, labelCol):
    x, y = ML_read_excel_No_split(fileName, featuresCol, labelCol)
    x, scalerX = ML_標準化1_轉換(x)
    y=y.reshape(y.shape[0],1)
    y, scalerY = ML_標準化1_轉換(y)
    y=y.reshape(y.shape[0])
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.05)
    return train_x, test_x, train_y, test_y,scalerX,scalerY


def ML_read_dataframe_標準化(fileName, featuresCol, labelCol):
    x, y = ML_read_excel_No_split(fileName, featuresCol, labelCol)
    x, scaler = ML_標準化1_轉換(x)
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.05)
    return train_x, test_x, train_y, test_y,scaler

# 讀取資料，有沒猜分
def ML_read_excel_No_split(fileName, featuresCol, labelCol):
    df = pd.read_excel(fileName)  # 讀取 pandas資料
    print(df.columns)  # 印出所有列
    x = df[featuresCol]  # 設定x的資料
    print(x)  # 印出x的資料
    y = df[labelCol]  # 設定y的資料
    print(y)
    ###   Pandas 轉 numpy
    x = x.to_numpy()  # x從 pandas 轉 numpy (參考Day34-524)
    print(x)  # 印出 轉 numpy後結果

    y = y.to_numpy()  # y從 pandas 轉 numpy
    print(y)
    print("資料筆數為", y.shape)
    y = y.reshape(y.shape[0])  # 將 y=y.to_numpy() 二維陣列,改為一維陣列(參考Day37-573)
    print(y, "外型大小")  # 印出結果
    return x, y


def ML_read_excel(fileName, featuresCol, labelCol):
    df = pd.read_excel(fileName)  # 讀取 pandas資料
    print(df.columns)  # 印出所有列
    x = df[featuresCol]  # 設定x的資料
    print(x)  # 印出x的資料
    y = df[labelCol]  # 設定y的資料
    print(y)
    ###   Pandas 轉 numpy
    x = x.to_numpy()  # x從 pandas 轉 numpy (參考Day34-524)
    print(x)  # 印出 轉 numpy後結果

    y = y.to_numpy()  # y從 pandas 轉 numpy
    print(y)
    print("資料筆數為", y.shape)
    y = y.reshape(y.shape[0])  # 將 y=y.to_numpy() 二維陣列,改為一維陣列(參考Day37-573)
    print(y, "外型大小")  # 印出結果
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.05)
    return train_x, test_x, train_y, test_y
    # #印出y的資料


def ML_read_CSV(fileName, featuresCol, labelCol):
    df = pd.read_csv(fileName)  # 讀取 pandas資料
    print(df.columns)  # 印出所有列
    x = df[featuresCol]  # 設定x的資料
    print(x)  # 印出x的資料
    y = df[labelCol]  # 設定y的資料
    print(y)
    ###   Pandas 轉 numpy
    x = x.to_numpy()  # x從 pandas 轉 numpy (參考Day34-524)
    print(x)  # 印出 轉 numpy後結果

    y = y.to_numpy()  # y從 pandas 轉 numpy
    print(y)
    print("資料筆數為", y.shape)
    y = y.reshape(y.shape[0])  # 將 y=y.to_numpy() 二維陣列,改為一維陣列(參考Day37-573)
    print(y, "外型大小")  # 印出結果
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.05)
    return train_x, test_x, train_y, test_y
    # #印出y的資料
from sklearn.metrics import mean_squared_error, mean_absolute_error

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

def ML_Regression_PolynomialFeaturesLinearRegression(train_x, test_x, train_y, test_y):
    print("======Polynomial Features Linear Regression======")
    poly_model = make_pipeline(PolynomialFeatures(4),
                               LinearRegression())  # poly_model 模型 make_pipeline(PolynomialFeatures(7), LinearRegression()) 七次方 線性回歸

    poly_model.fit(train_x, train_y)
    xfit = test_x  # 0 到 10 之間 均勻的產生 1000個點
    yfit = poly_model.predict(test_x)  # 預測 Y值    X值 使用 0 到 10 之間 均勻的產生 1000個點

    print('MAE:', mean_absolute_error(yfit, test_y))
    print('MSE:', mean_squared_error(yfit, test_y))

    print("0-7次方的各項係數: ", poly_model.named_steps["linearregression"].coef_)  # 印出 0-7 次方的各項係數:

    plt.scatter(train_x[:,2], train_y)  # 把 x 和 y 的分布在圖表上顯示
    plt.scatter(xfit[:,2], yfit)  # 把 一千個x值 和 預測出來的Y值 的線圖在圖表上顯示
    plt.legend(['Train Data', 'poly_model  Predictions'])
    plt.savefig("PolynomialFeaturesLinearRegression.jpg")
    plt.show()

from sklearn.svm import SVR
def ML_Regression_SVR(train_x, test_x, train_y, test_y):
    print("======SVR======")
    model = SVR(shrinking=True, kernel='poly', gamma=0.01, epsilon=1, degree=3, C=0.1)
    model.fit(train_x, train_y)
    yfit = model.predict(test_x)

    print('MAE:', mean_absolute_error(yfit, test_y))
    print('MSE:', mean_squared_error(yfit, test_y))

    plt.scatter(train_x[:,2], train_y)  # 把 x 和 y 的分布在圖表上顯示
    plt.scatter(test_x[:,2], yfit)  # 把 一千個x值 和 預測出來的Y值 的線圖在圖表上顯示
    plt.legend(['Train Data', 'SVR  Predictions'])
    plt.savefig("SVR.jpg")
    plt.show()

from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LinearRegression, BayesianRidge
def ML_Regression_BayesianRidgePolynomialRegression(train_x, test_x, train_y, test_y):
    # bayesian ridge polynomial regression
    print("======bayesian ridge polynomial regression ======")

    tol = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
    alpha_1 = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
    alpha_2 = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
    lambda_1 = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
    lambda_2 = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
    normalize = [True, False]

    bayesian_grid = {'tol': tol, 'alpha_1': alpha_1, 'alpha_2': alpha_2, 'lambda_1': lambda_1, 'lambda_2': lambda_2,
                     #'normalize': normalize
                     }

    bayesian = BayesianRidge(fit_intercept=False)
    bayesian_search = RandomizedSearchCV(bayesian, bayesian_grid, scoring='neg_mean_squared_error', cv=3,
                                         return_train_score=True, n_jobs=-1, n_iter=40, verbose=1)
    bayesian_search.fit(train_x, train_y)

    print(bayesian_search.best_params_)

    bayesian_confirmed = bayesian_search.best_estimator_
    text_x_predict = bayesian_confirmed.predict(test_x)
    #bayesian_pred = bayesian_confirmed.predict(bayesian_poly_future_forcast)
    print('MAE:', mean_absolute_error(text_x_predict, test_y))
    print('MSE:', mean_squared_error(text_x_predict, test_y))

    plt.plot(test_y)
    plt.plot(text_x_predict)
    plt.legend(['Test Data', 'Bayesian Ridge Polynomial Predictions'])
    plt.show()


def ML_Classification_RandomForestClassifier(train_x, test_x, train_y, test_y):
    # 隨機森林

    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(n_estimators=100, max_depth=10,
                                random_state=2)
    rf.fit(train_x, train_y.ravel())
    prediction = rf.predict(test_x)
    rfScore = rf.score(test_x, test_y)
    print("隨機森林 預估答案      ：", prediction, " 準確率：", rfScore)
    ###
    from sklearn.tree import export_graphviz
    export_graphviz(rf.estimators_[2], out_file='隨機森林1.dot',
                    # feature_names=colX,
                    # class_names =colY2,
                    rounded=True, proportion=False,
                    precision=2, filled=True)

    #######
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

def ML_Classification_DecisionTreeClassifier(train_x, test_x, train_y, test_y):
    # 決策樹

    from sklearn import tree
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(train_x, train_y.ravel())
    prediction = clf.predict(test_x)
    clfScore = clf.score(test_x, test_y)
    print("決策樹 預估答案       ：", prediction, " 準確率：", clfScore)

    #####
    tree.export_graphviz(clf, out_file='決策樹.dot')
    # 換成中文的字體
    # plt.rcParams['font.新細明體'] = ['SimSun'] # 步驟一（替換sans-serif字型）
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
    plt.rcParams['axes.unicode_minus'] = False  # 步驟二（解決座標軸負數的負號顯示問題）
    plt.rcParams.update({'font.size': 8})
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

def ML_Classification_KNeighborsClassifier(train_x, test_x, train_y, test_y):
    # KNN 演算法

    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors=3, p=1)
    knn.fit(train_x, train_y)
    knnPredict = knn.predict(test_x)
    knnScore = knn.score(test_x, test_y)
    print("KNN    演算法 預估答案：", knnPredict, " 準確率：", knnScore)

from sklearn import tree
def ML_Classification_DecisionTreeClassifier_gini(train_x, test_x, train_y, test_y):
    # 決策樹 演算法

    import pydot
    # https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier
    clf = tree.DecisionTreeClassifier(criterion='gini')
    clf = clf.fit(train_x, train_y)
    tree.export_graphviz(clf, out_file='tree-C1.dot')
    clfPredict = clf.predict(test_x)
    clfScore1 = clf.score(test_x, test_y)
    print("決策樹 1 演算法 預估答案：", clfPredict, " 準確率：", clfScore1)

def ML_Classification_DecisionTreeClassifier_entropy(train_x, test_x, train_y, test_y):
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
                       feature_names=colX,
                       # class_names=colY,
                       filled=True)

    fig.savefig("decistion_tree.png")
    plt.show()

"""
p 內定值=2
Euclidean Distance
距離＝  √((x1-x2)**2+(y1-y2)**2)


p =1
Manhattan Distance
距離＝｜x1-x2｜+|y1-y2|
"""
def ML_分類_KNN(train_x, test_x, train_y, test_y, k=5, p=1):
    knn = KNeighborsClassifier(n_neighbors=k, p=p)
    knn.fit(train_x, train_y)
    print("預測", knn.predict(test_x))
    print("實際", test_y)
    print('準確率: %.2f' % knn.score(test_x, test_y))
    return knn


def ML_群聚_KMeans(train_x, test_x, train_y, test_y,k=0):
    # KMeans 演算法
    if(k==0):
        kmeans = KMeans()
    else:
        kmeans = KMeans(n_clusters=k)
    kmeans.fit(train_x)

    print("實際", test_y)
    print("預測", kmeans.predict(test_x))
    print('準確率:%.3f' % metrics.accuracy_score(test_y, kmeans.predict(test_x)))

    # Storing the predicted Clustering labels
    labels = kmeans.predict(test_x)
    # Evaluating the performance
    print("修正後的準確率: %.3f" % metrics.homogeneity_score(test_y, labels))
    print("Homogeneity: %0.3f" % metrics.homogeneity_score(test_y, labels))
    print("Completeness: %0.3f" % metrics.completeness_score(test_y, labels))
    print("V-measure: %0.3f" % metrics.v_measure_score(test_y, labels))

    # Evaluate the score 修正答案的對照表
    hscore = metrics.homogeneity_score([0, 1, 0, 1], [1, 0, 1, 0])
    print(hscore)
    centers = kmeans.cluster_centers_
    print("中心點：",centers)
    return kmeans


from sklearn.preprocessing import MinMaxScaler


# train_x, scaler=myfun.ML_標準化1_轉換(train_x)
def ML_標準化1_轉換(x):
    scaler = MinMaxScaler()  # 初始化
    scaler.fit(x)  # 找標準化範圍
    x1 = scaler.transform(x)  # 把資料轉換
    return x1, scaler


def ML_標準化1_還原(x1, scaler):
    print(x1)
    print("還原")
    x2 = scaler.inverse_transform(x1)
    print(x2)
    return x2


def ML_標準化1_還原Y(test_y, scalerY):
    test_y3=scalerY.inverse_transform(test_y.reshape(test_y.shape[0],1)).reshape(test_y.shape[0])
    return test_y3


def ML_標準化2_轉換(x):
    dict1 = {}
    min1 = np.min(x, axis=0)
    dict1["min"] = min1
    max1 = np.max(x, axis=0)
    dict1["max"] = max1
    dist = max1 - min1
    dict1["dist"] = dist
    #  x-最低/(最高-最低)
    x2 = (x - min1) / dist
    return x2, dict1


def ML_標準化2_還原(x1, dict1):
    min1 = dict1["min"]
    max1 = dict1["max"]
    dist = dict1["dist"]
    #  (x1*(最高-最低))+最低
    x2 = (x1 * (dist)) + min1
    return x2


from matplotlib.font_manager import FontProperties  # 中文字體


def matplot_中文字():
    # 換成中文的字體
    # plt.rcParams['font.新細明體'] = ['SimSun'] # 步驟一（替換sans-serif字型）
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
    plt.rcParams['axes.unicode_minus'] = False  # 步驟二（解決座標軸負數的負號顯示問題）






def seaborn_all(df,title = "title",x = "Age",y = "ALB",hue = "Sex",style = "Category",col = "Category"):
    seaborn_依照種類繪圖(df, x=None, y="ALB", hue="Category")
    plt.savefig("seaborn_依照種類繪圖" + str(x) + "_" + str(y) + ".jpg")

    #matplotlib_heatmap_未完成(df, title=title)
    #plt.savefig("matplotlib_heatmap" + str(x) + "_" + str(y) + ".jpg")
    #matplotlib_heatmap_顏色值_未完成(df, title=title)
    #plt.savefig("matplotlib_heatmap_顏色值_未完成" + str(x) + "_" + str(y) + ".jpg")
    #seaborn_蠟燭圖_未完成(df, x="Age", y="ALB", hue="Category")
    #plt.savefig("seaborn_蠟燭圖_未完成" + str(x) + "_" + str(y) + ".jpg")

    ###################################
    seaborn_replot_point(df, x=x, y=y, hue=hue, style=style, col=None)
    plt.savefig("seaborn_replot_point1" + str(x) + "_" + str(y) + ".jpg")
    seaborn_replot_point(df, x=x, y=y, hue=hue, style=style, col=None, marker="x")
    plt.savefig("seaborn_replot_point2" + str(x) + "_" + str(y) + ".jpg")
    seaborn_replot_point(df, x=x, y=y, hue=hue, style=style, col=col)
    plt.savefig("seaborn_replot_point3" + str(x) + "_" + str(y) + ".jpg")
    seaborn_replot_line(df, x=x, y=y, hue=hue, style=style, col=col)
    plt.savefig("seaborn_replot_line" + str(x) + "_" + str(y) + ".jpg")
    seaborn_replot_line_range(df, x=x, y=y, hue=hue, style=style, col=col)
    plt.savefig("seaborn_replot_line_range" + str(x) + "_" + str(y) + ".jpg")
    seaborn_直線(df, x=x, y=y, hue=hue, style=style, col=col)
    plt.savefig("seaborn_直線" + str(x) + "_" + str(y) + ".jpg")
    seaborn_柱狀圖上下(df, x=x, y=y, hue=hue, style=style, col=col)
    plt.savefig("seaborn_柱狀圖上下" + str(x) + "_" + str(y) + ".jpg")
    seaborn_直線橫線(df, x=x, y=y, hue=hue, style=style, col=col, kind="hist")
    plt.savefig("seaborn_直線橫線" + str(x) + "_" + str(y) + ".jpg")
    seaborn_上下點(df, x=x, y=y, hue=hue, style=style, col=col, kind="hist")
    plt.savefig("seaborn_上下點" + str(x) + "_" + str(y) + ".jpg")
    seaborn_海帶上下(df, x=x, y=y, hue=hue, style=style, col=col)
    plt.savefig("seaborn_海帶上下1" + str(x) + "_" + str(y) + ".jpg")
    seaborn_海帶上下(df, x=x, y=y, hue=hue, style=None, col=None)
    plt.savefig("seaborn_海帶上下2" + str(x) + "_" + str(y) + ".jpg")
    seaborn_直柱上下平排(df, x=x, y=y, hue=hue, style=None, col=None)
    plt.savefig("seaborn_直柱上下平排" + str(x) + "_" + str(y) + ".jpg")
    seaborn_直柱上下(df, x=x)
    plt.savefig("seaborn_直柱上下" + str(x) + "_" + str(y) + ".jpg")
    seaborn_直柱左右(df, x=x)
    plt.savefig("seaborn_直柱左右" + str(x) + "_" + str(y) + ".jpg")
    seaborn_二筆資料交叉分析(df, x=x, y=y, hue=hue)
    plt.savefig("seaborn_二筆資料交叉分析" + str(x) + "_" + str(y) + ".jpg")
    seaborn_多筆資料交叉分析(df, hue=hue)
    plt.savefig("seaborn_多筆資料交叉分析" + str(x) + "_" + str(y) + ".jpg")
    seaborn_多筆資料交叉分析_等高線(df, hue=hue)
    plt.savefig("seaborn_多筆資料交叉分析_等高線" + str(x) + "_" + str(y) + ".jpg")


def seaborn_replot_point(df,x="Age", y="ALB", hue="Category", style="Category",col="",marker=None):
    sns.relplot(
        data=df,
        x=x, y=y,
        marker=marker,
        hue=hue, style=style, col=col )

def seaborn_replot_line(df, x="Age", y="ALB", hue="Category", style="Category",col=""):
    sns.relplot(
        data=df,
        x=x, y=y,
        hue=hue, style=style,  kind="line", col=col,
        # col="align",
        # size="coherence",
        # style="choice",
        facet_kws=dict(sharex=False),
    )

def seaborn_replot_line_range(df, x="Age", y="ALB", hue="Category", style="Category",col=""):
    sns.relplot(
        data=df,
        x=x, y=y,
        hue=hue, style=style,
        kind="line", col=col,
    )

def seaborn_直線(df, x="Age", y="ALB", hue="Category",  style="Category", col=""):
    sns.lmplot(  data=df,
        x=x, y=y,
        hue=hue,col=col )

def seaborn_柱狀圖上下(df, x="Age", y="ALB", hue="Category",style="Category",   col=""):
    try:
        sns.displot(data=df,x=x, y=y, col=col, kde=True)
    except:
        print("error: seaborn_replot_柱狀圖上下")


def seaborn_直線橫線(df, x="Age", y="ALB", hue="Category",style="Category",   col="",kind="ecdf"):

    try:
        sns.displot(data=df,
        x=x, y=y,
        hue=hue,col=col,
        kind=kind,  # 'hist', 'kde', 'ecdf'
        rug=True)
    except:
        print("error: seaborn_replot_柱狀圖上下")

def seaborn_上下點(df, x="Age", y="ALB", hue="Category", style="Category", col="", kind="ecdf"):
    try:
        sns.catplot(
            data=df,
            x=x, y=y,
            hue=hue, col=col,
            kind="swarm")
    except:
        print("error: seaborn_上下點")


def seaborn_海帶上下(df, x="Age", y="ALB", hue="Sex", style="Category", col=None):
    sns.catplot(
        data=df,
        x=x, y=y,
        col=col,
        hue=hue,
        kind="violin",
        split=True)

def seaborn_直柱上下平排(df, x="Age", y="ALB", hue="Category", style="Category", col=""):
    sns.catplot(
        data=df,
        x=x, y=y,
        hue=hue,
          kind="bar" )



def seaborn_直柱上下(df, x="Age", y="ALB", hue="Category", style="Category", col=""):
    sns.distplot(x=df[x])


def seaborn_直柱左右(df, x="Age", y="ALB", hue="Category", style="Category", col=""):
    sns.distplot(x=df[x])


def seaborn_二筆資料交叉分析(df, x="Age", y="ALB", hue="Category"):

    sns.jointplot(
        data=df,
        x=x, y=y,
        hue=hue,
    )


def seaborn_多筆資料交叉分析(df, hue="Category"):
    sns.pairplot(
        data=df,
        hue=hue,
    )



def seaborn_多筆資料交叉分析_等高線(df, hue="Category"):

    g = sns.PairGrid(data=df,
        hue=hue)
    g.map_diag(sns.kdeplot)
    g.map_offdiag(sns.kdeplot, cmap="Blues_d", n_levels=6)






def matplotlib_heatmap_未完成(df, title="Category"):
    # 換成中文的字體
    # plt.rcParams['font.新細明體'] = ['SimSun'] # 步驟一（替換sans-serif字型）
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
    plt.rcParams['axes.unicode_minus'] = False  # 步驟二（解決座標軸負數的負號顯示問題）


    #vegetables = df.columns
    #farmers =df.columns
    #harvest = df.to_numpy()
    vegetables = ["cucumber", "tomato", "lettuce", "asparagus",
                  "potato", "wheat", "barley"]
    farmers = ["Farmer Joe", "Upland Bros.", "Smith Gardening",
               "Agrifun", "Organiculture", "BioGoods Ltd.", "Cornylee Corp."]

    harvest = np.array([[0.8, 2.4, 2.5, 3.9, 0.0, 4.0, 0.0],
                        [2.4, 0.0, 4.0, 1.0, 2.7, 0.0, 0.0],
                        [1.1, 2.4, 0.8, 4.3, 1.9, 4.4, 0.0],
                        [0.6, 0.0, 0.3, 0.0, 3.1, 0.0, 0.0],
                        [0.7, 1.7, 0.6, 2.6, 2.2, 6.2, 0.0],
                        [1.3, 1.2, 0.0, 0.0, 0.0, 3.2, 5.1],
                        [0.1, 2.0, 0.0, 1.4, 0.0, 1.9, 6.3]])





    fig, ax = plt.subplots()
    im = ax.imshow(harvest)

    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(len(farmers)), labels=farmers)
    ax.set_yticks(np.arange(len(vegetables)), labels=vegetables)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(vegetables)):
        for j in range(len(farmers)):
            text = ax.text(j, i, harvest[i, j],
                           ha="center", va="center", color="w")

    ax.set_title(title)
    fig.tight_layout()



def seaborn_蠟燭圖_未完成(df, x="Age", y="ALB", hue="Category"):
    # Random test data
    np.random.seed(19680801)
    all_data = [np.random.normal(0, std, size=100) for std in range(1, 4)]
    labels =   ['x1', 'x2', 'x3']


    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(9, 4))

    # rectangular box plot
    bplot1 = ax1.boxplot(all_data,
                         vert=True,  # vertical box alignment
                         patch_artist=True,  # fill with color
                         labels=labels)  # w

    # notch shape box plot
    bplot2 = ax2.boxplot(all_data,
                         notch=True,  # notch shape
                         vert=True,  # vertical box alignment
                         patch_artist=True,  # fill with color
                         labels=labels)  # will be used to label x-ticks
    ax2.set_title('Notched box plot')

    # fill with colors
    colors = ['pink', 'lightblue', 'lightgreen']
    for bplot in (bplot1, bplot2):
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)

    # adding horizontal grid lines
    for ax in [ax1, ax2]:
        ax.yaxis.grid(True)
        ax.set_xlabel('Three separate samples')
        ax.set_ylabel('Observed values')





def seaborn_依照種類繪圖(df, x=None, y="ALB", hue="Category"):
    fig, ax = plt.subplots()
    values = df[hue].unique()
    for i in values:
        df0 = df[df[hue] == i]
        if x==None:
            df0 = df0.sort_values(y)
            x1 = range(df0[y].shape[0])
        else:
            x1=df0[x]
        label = hue + "=" + str(i)
        ax.plot(x1, df0[y], label=label)
    plt.legend()
    plt.title(y)

"""
https://matplotlib.org/stable/gallery/statistics/boxplot_vs_violin.html#sphx-glr-gallery-statistics-boxplot-vs-violin-py
https://matplotlib.org/stable/gallery/pie_and_polar_charts/bar_of_pie.html#sphx-glr-gallery-pie-and-polar-charts-bar-of-pie-py
https://matplotlib.org/stable/gallery/pie_and_polar_charts/pie_features.html#sphx-glr-gallery-pie-and-polar-charts-pie-features-py
https://matplotlib.org/stable/gallery/animation/animate_decay.html#sphx-glr-gallery-animation-animate-decay-py
https://matplotlib.org/stable/gallery/animation/random_walk.html#sphx-glr-gallery-animation-random-walk-py
https://matplotlib.org/stable/gallery/animation/unchained.html#sphx-glr-gallery-animation-unchained-py
https://matplotlib.org/stable/gallery/mplot3d/2dcollections3d.html#sphx-glr-gallery-mplot3d-2dcollections3d-py
https://seaborn.pydata.org/generated/seaborn.kdeplot.html
https://codertw.com/%E7%A8%8B%E5%BC%8F%E8%AA%9E%E8%A8%80/736148/





"""