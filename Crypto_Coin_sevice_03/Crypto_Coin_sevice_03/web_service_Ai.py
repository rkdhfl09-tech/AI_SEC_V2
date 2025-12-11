import base64
import numpy as np
from flask import Flask,render_template,request,json,jsonify
from ai_service.Crypto_Coin_service import input_request
from movie_review.Naver_NLP_service import getPredict
#NLP 임포트
import movie_review.Naver_NLP_Predict as nnp
app = Flask(__name__)
#**  after  코인명 추가시 리스트 추가
COIN_NAMES = ["BTC","ETH","XRP"]
COIN_HAN = ["비트코인","이더리움","리플"]
AI_PATH = "ai_service/"
#utils
#코인 가격 분석 모델 연결과 응답
def crypto_coin_anal(coinname,timegap):
    #모델 호출과 분석 결과 리턴
    return input_request(coinname, timegap)
#routes
@app.route("/")# 메인 인트로 페이지
def root():
    return render_template("intro_index.html")
@app.route("/page/<pagename>")
def page_href(pagename):
    return render_template(f"{pagename}.html")
@app.route("/coin_name")
def out_coinname():
    coin_name_dict = {"eng_name":COIN_NAMES,
                      "han_name":COIN_HAN}
    return jsonify(coin_name_dict)
@app.route("/user_data",methods=["POST"])#코인 가격 예측 분석 페이지
def user_data():
    user_datas = request.get_json()
    print(user_datas)
    coinname=user_datas["coinname"]
    timegap=int(user_datas["timegaps"])
    report = crypto_coin_anal(coinname,timegap)
    print(report)
    return jsonify(report)
@app.route("/page/review_main")
def ret_mainpage():
    return  render_template("nlp_review.html")
@app.route("/movie_review",methods=["POST"])
def movie_review_pred():
    user_data = request.get_json()
    print(user_data["send_text"])
    return ""
    #return getPredict(user_data)
app.run("127.0.0.1",4321,True)