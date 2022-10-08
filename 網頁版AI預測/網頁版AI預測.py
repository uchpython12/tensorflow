import socketserver as socketserver
from http.server import SimpleHTTPRequestHandler as RequestHandler
from urllib.parse import urlparse
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import model_from_json

category=3
dim=4
x_test = np.array([[5.1,3.5,1.4,0.2],[7.0,3.2,4.7,1.4],[6.9,3.1,5.4,2.1],[5.9,3.0,5.1,1.8]])
y_test =  np.array([0,1,2,2])
y_test2=tf.keras.utils.to_categorical(y_test, num_classes=(category))

# 讀取模型架構
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# 讀取模型權重
model.load_weights("model.h5")
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss = tf.keras.losses.categorical_crossentropy,
    metrics = ['accuracy'])

#######

class MyHandler(RequestHandler):          # 物件導向OOP
    def do_GET(self):                     # GET 的方法
        global model
        t1=urlparse(self.path)
        path = urlparse(self.path).path          # 取得路徑
        if path != '/':
            # 其他  例如：http://127.0.0.1:8888/1.jpg
            super(MyHandler, self).do_GET()   # 呼叫父類別的方法
        else:
            # 只處理  http://127.0.0.1:8888/
            # 只處理  http://127.0.0.1:8888/?name=xxx&password=xxx

            dict1={}
            t1=urlparse(self.path)
            # get path and file name
            t2=urlparse(self.path).path
            query = urlparse(self.path).query  # 取得和解析網路完整的URL
            if query != "":
                # ?name=a&password=b  轉  {"name":"a","password":"b"}
                dict1 = dict(qc.split("=") for qc in query.split("&"))

            # 定義網路的資料型態
            self.send_response(200)           # 回傳200 網路定義
            self.send_header("Content-type", "text/html; charset=utf-8")  # 回傳 資料型態HTML
            self.end_headers()

            para1=dict1.get("para1", '沒有這個資料')
            para2= dict1.get("para2", '沒有這個資料')
            para3= dict1.get("para3", '沒有這個資料')
            para4= dict1.get("para4", '沒有這個資料')

            str1 = "花瓣長度="+para1+\
                   "<br> 花瓣寬度="+para2+\
                   " <br>花萼長度="+para3+\
                   " <br>花萼寬度="+para4
            if para1!="沒有這個資料":
                para1=float(para1)
                para2=float(para2)
                para3=float(para3)
                para4=float(para4)


                x_test=np.array([[para1,para2,para3,para4]])
                predict = model.predict(x_test)
                ans=np.argmax(predict, axis=-1)
                str1=str1+"<br>答案是"+str(ans)+\
                     "<br>屬於每個答案的機率為"+str(predict)


            html1=str1.encode("utf-8")               # 把字串 轉成 bytes
            self.wfile.write(html1)           # 把資料 回傳到  瀏覽器

socketserver.TCPServer.allow_reuse_address = True    # 可以重複使用IP
httpd = socketserver.TCPServer(('0.0.0.0', 8888), MyHandler)  # 啟動WebServer
try:
    httpd.serve_forever()                          # 等待用戶使用 WebServer
except:
    print("Closing the server.")
    httpd.server_close()                           # 關閉 WebServer