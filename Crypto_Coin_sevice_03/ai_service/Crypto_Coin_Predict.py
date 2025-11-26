import tensorflow as tf
import numpy as np
import matplotlib.pyplot
import requests
from ai_service.Crypto_Coin_Train import get_data, extract_data
import pickle

TIME_STEP = 60
CUR = 0  # 현재가 인덱스
HIGH = 1  # 최고가 인덱스
LOW = 2  # 최저가 인덱스
import os

COIN_PATH = os.path.join(os.path.dirname(__file__), "coin_config")

#  데이터 수신함수
#  예측데이터생성
robust_scaler = None


def create_predict(x_pred):
    # print(x_pred.shape)#(200, 3)
    x_pred = x_pred[::-1]  # 시간오름차순 정렬
    return x_pred[-TIME_STEP:, :]  # 맨뒤 8개 데이터 추출 리턴


#  모델 불러오기
def load_rnnmodel(tpath):
    return tf.keras.models.load_model(tpath, compile=True)


#  일자에 따른 예측값 출력함수
def y_predict(model, x_pred, timegap, coinname="BTC"):
    global robust_scaler
    data_array = []
    if not robust_scaler:
        with open(f"{COIN_PATH}/{coinname.lower()}_robust_scaler", "rb") as fp:
            robust_scaler = pickle.load(fp)
    for i in range(timegap):
        y_pred = model.predict(x_pred)
        y_true = robust_scaler.inverse_transform(y_pred)
        data_array.append(y_true[0].tolist())
        x_pred = x_pred[:, 1:, :]
        print(x_pred.shape)  # (1, 59, 3)
        print(y_pred.shape)  # (1,3)
        y_pred = y_pred.reshape(1, 1, 3)
        print(y_pred.shape)
        x_pred = np.concatenate((x_pred, y_pred), axis=1)
        print("데이터정합성 검증:", y_pred[0][0][0] == x_pred[0][-1][0])
    return data_array


def convert_price(price_data, coinname):
    global robust_scaler
    if not robust_scaler:
        with open(f"{COIN_PATH}/{coinname}_robust_scaler", "rb") as fp:
            robust_scaler = pickle.load(fp)
    return robust_scaler.inverse_transform(price_data)


#  예측값 가격 복원 함수


if __name__ == "__main__":

    print("실행 시점 테스트")
    coin_name = "BTC"
    raw_data = get_data(f"https://api.bithumb.com/v1/candles/days?market=KRW-{coin_name}&count=200")
    # print("=== 현재 원형 가격 확인 1 =========")
    # print("현재가 ",raw_data[0]["trade_price"])
    # print("최고가 ",raw_data[0]["high_price"])
    # print("최저가 ",raw_data[0]["low_price"])

    preproc_data = extract_data(raw_data, False)
    # cur_1=convert_price(preproc_data)[0]
    # print("=== 변형 가격 확인 2 =========")
    # print("현재가 ",cur_1[0])
    # print("최고가 ",cur_1[1])
    # print("최저가 ",cur_1[2])

    x_pred = create_predict(preproc_data)
    # cur_2=convert_price(x_pred)[-1]
    # print("=== 정렬후 가격 확인 3 =========")
    # print("현재가 ",cur_2[0])
    # print("최고가 ",cur_2[1])
    # print("최저가 ",cur_2[2])

    print(len(x_pred))
    print(x_pred.shape)
    print(x_pred[0])
    lstm_model = load_rnnmodel(f"{COIN_PATH}/lstm_model.keras")
    print(lstm_model)
    y_predarr = y_predict(lstm_model, np.array([x_pred]), 7)
    cur_price = convert_price(x_pred)
    print("오늘현재가격:", cur_price[-1][0])
    print("오늘최고가격:", cur_price[-1][1])
    print("오늘최저가격:", cur_price[-1][2])
    day_cnt = 1
    for curp, highp, lowp in y_predarr:
        print(f"{day_cnt} 일차: 기준가:{curp} 최고가:{highp} 최저가:{lowp}")
        day_cnt += 1

# In[ ]:



