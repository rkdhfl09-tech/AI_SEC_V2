import base64

from flask import Flask,render_template,request,json,jsonify
app = Flask(__name__)
#주소 라우팅
@app.route("/")
def root_connect():
    return render_template("index.html")
@app.route("/myinfo",methods=["post"])
def myinformation():
    myname=(request.form["myname"])
    age=(request.form["age"])
    return render_template("join.html",myname=myname,age=age)
#base 64 인코딩을 이용하여 파일원본을 전송
@app.route("/getimg/<imgname>")
def getimage(imgname="tiger.jpg"):
    with open(f"static/img/{imgname}","rb") as fp:
        img_byte = fp.read()
        #이미지를 byte 값으로 읽어들어 아스키 인코딩후 완성형문자로 전송과정
        encoded = base64.b64encode(img_byte).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"

@app.route("/aaa")
def aaa():
    return "aaa 쿼리스트링을 가지고 접속하였습니다."
#주소로 데이터를 수신하는 방법
@app.route("/bbb/<name>",methods=["get"])
@app.route("/ccc/<name>",methods=["get"])
def get_param(name):
    print("bbb에 데이터를 보냈습니다",name)
    return f"{name}님 환영합니다."
#주소 쿼리스트링으로 데이터 수신방법
@app.route("/fff")
def get_querystring():
    name=(request.args.get("name"))
    age=(request.args.get("age"))
    return f"당신의 이름은 {name} 나이는 {age}입니다."
#Post 방식의 데이터 수신방법
@app.route("/jtest")
def jsondata_test():
    return render_template("jtest.html")
@app.route("/jtest/jdata",methods=["post"])
def get_jdata():
    print("jso 진입======")
    jdict = (request.get_json())
    print(jdict["myname"])
    print(jdict["age"])
    return jsonify(jdict)




app.run("127.0.0.1",4321,True)
