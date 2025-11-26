import tensorflow as tf
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import requests
import random
import pickle
import os
from tensorflow.keras import Input, Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, BatchNormalization

print(tf.__version__)
random.seed(111)  # 일률적인 학습을 위함
np.random.seed(111)
tf.random.set_seed(111)
# 0. 환경변수
TIME_STEP = 60
CUR = 0  # 현재가 인덱스
HIGH = 1  # 최고가 인덱스
LOW = 2  # 최저가 인덱스
COIN_PATH = \
    os.path.join(os.path.dirname(__file__),
                 "coin_config")


# 1. 데이터 수집
def get_data(url):
    raw_data = requests.get(url).json()
    return raw_data


def extract_data(raw_data, new_data=True, coin_name="btc"):
    # comment : 필요한 데이터 추출(time_stemp,trade_price,high_price,low_price)과 사분법 실행
    data_sets = []
    for i in range(len(raw_data)):
        unit_arr = []
        unit_arr.append(raw_data[i]["trade_price"])
        unit_arr.append(raw_data[i]["high_price"])
        unit_arr.append(raw_data[i]["low_price"])
        data_sets.append(unit_arr)
    data_sets = np.array(data_sets, dtype=np.float64)  # (200, 3)
    print(data_sets.shape)
    print(data_sets.max())
    print(data_sets.min())
    # 비이상치 데이터가 존재할때 쓰는 전처리 기법이다.
    robust_scaler = None
    if new_data:
        robust_scaler = sklearn.preprocessing.RobustScaler()  # 1차 사분법, 중앙값기준 정규화
        data_sets = robust_scaler.fit_transform(data_sets)
        if not os.path.exists(COIN_PATH):
            os.makedirs(COIN_PATH)
        with open(f"{COIN_PATH}/{coin_name.lower()}_robust_scaler", "wb") as fp:
            pickle.dump(robust_scaler, fp)
    else:
        with open(f"{COIN_PATH}/{coin_name.lower()}_robust_scaler", "rb") as fp:
            robust_scaler = pickle.load(fp)
        data_sets = robust_scaler.transform(data_sets)
    # print(data_sets.max())
    # print(data_sets.min())
    # print(data_sets.shape)
    return data_sets


def rnn_data_create(preproc_data):
    preproc_data = preproc_data[::-1]  # 역순정렬
    # (200, 4)
    x_data = []
    y_data = []
    for i in range(len(preproc_data) - TIME_STEP):
        x_data.append(preproc_data[i:i + TIME_STEP])
        y_data.append(preproc_data[i + TIME_STEP])
    # 최근 데이터를 뒤로 배치
    return np.array(x_data), np.array(y_data)


def struct_model():
    model = Sequential()
    model.add(Input((TIME_STEP, 3)))
    lstm_1 = tf.keras.layers.LSTM(
        64,
        activation='tanh',
        recurrent_activation='sigmoid',
        return_sequences=True)
    lstm_2 = tf.keras.layers.LSTM(
        32,
        activation='tanh',
        recurrent_activation='sigmoid',
        return_sequences=True)
    lstm_3 = tf.keras.layers.LSTM(
        16,
        activation='tanh',
        recurrent_activation='sigmoid',
        return_sequences=False)
    model.add(lstm_1)
    model.add(lstm_2)
    model.add(lstm_3)
    model.add(BatchNormalization())
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.4))
    model.add(Dense(64, activation="relu"))
    model.add(Dense(16, activation="relu"))
    model.add(Dense(3, activation="linear"))
    model.compile(loss="mse", optimizer="adam", metrics=["mae"])
    return model


def samle_weight(length):  # 시간 흐름에 따른 가중치 부여
    return np.linspace(0.0001, 1, length)


def train_fit(tmodel, x_train, y_train, epoch, time_weight):  # 훈련 실행
    cb = tf.keras.callbacks.EarlyStopping(  # 조기종료 기법 사용
        monitor='val_loss',  # 체크할 값
        patience=50,  # 체크후 훈련 진행한도
        verbose=1,  # 화면 표기
        restore_best_weights=True  # 최적의 가중치로 복원
        # start_from_epoch=400# 이후 훈련부터 체크 실행
    )
    return tmodel.fit(x_train, y_train, validation_data=(x_train, y_train), epochs=epoch,
                      batch_size=len(x_train) // 10, sample_weight=time_weight, callbacks=[cb],
                      verbose=0)


def result_graph(fit_history, coin_name):
    history = fit_history.history
    plt.subplot(1, 2, 1)
    plt.plot(history["loss"], label="train_loss")
    plt.plot(history["val_loss"], label="valid_loss")
    plt.title("MSE")
    plt.subplot(1, 2, 2)
    plt.plot(history["mae"], label="train_mae")
    plt.plot(history["val_mae"], label="valid_mae")
    plt.title("MAE")
    if not os.path.exists(f"{COIN_PATH}/static"):
        os.makedirs(f"{COIN_PATH}/static/chart")
    plt.savefig(r"{}/static/chart/{}_mse_mae.png".format(COIN_PATH, coin_name.lower()))
    plt.close()


def confirm_pred(y_true, y_pred, coin_name):
    if y_true.shape == y_pred.shape:
        plt.plot(y_true, y_true, label="Y_TRUE")
        plt.scatter(y_true, y_pred, s=2, color="red", label="Y_PRED")
        if not os.path.exists(f"{COIN_PATH}/static"):
            os.makedirs(f"{COIN_PATH}/static/chart")
        plt.savefig(r"{}/static/chart/{}_scatter.png".format(COIN_PATH, coin_name.lower()))
        plt.close()


if __name__ == "__main__":
    coin_names = ["BTC", "ETH", "XRP"]
    for coin_name in coin_names:
        raw_data = get_data(f"https://api.bithumb.com/v1/candles/days?market=KRW-{coin_name}&count=200")
        # print("데이터 수량 출력:",len(raw_data))
        # print("데이터 샘플 출력:",raw_data[0])
        # print("데이터 키값 출력:",raw_data[0].keys())
        preproc_data = extract_data(raw_data, coin_name=coin_name)
        x_train, y_train = rnn_data_create(preproc_data)
        # 데이터 정합성
        # print(y_train[0]==x_train[1][-1])
        # print(y_train[1]==x_train[2][-1])

        # 데이터 형태 검증
        # print(type(x_train[0][0][0]))
        # print(type(x_train[0][0][1]))
        # print(type(x_train[0][0][2]))
        # print(type(y_train[0][0]))
        # 데이터 확인
        # print((x_train[0][0][0]))
        # print((x_train[0][0][1]))
        # print((x_train[0][0][2]))
        # print((y_train[0][0]))
        model = struct_model()
        # 모델 작동 확인
        # res = model(x_train)
        # print(res.shape)
        time_weight = samle_weight(len(x_train))
        # print(x_train.shape)
        # print(y_train.shape)
        # print(len(time_weight))
        print(coin_name + " 훈련중 ............")
        fit_history = train_fit(model, x_train, y_train, 500, time_weight)
        result_graph(fit_history, coin_name)
        y_pred = model.predict(x_train)
        # print(y_train.shape)
        # print(y_pred.shape)
        confirm_pred(y_train, y_pred, coin_name)

        # 오차율 산정
        # y_pred , y_train
        print(y_pred.shape)
        print(y_train.shape)
        # 현재가 오차율
        y_gap = y_train - y_pred
        print(y_gap.shape)
        # 절대값 변경
        y_abs_gap = np.absolute(y_gap)
        print((y_abs_gap < 0).sum())
        y_mean = np.mean(y_abs_gap, axis=0)
        print(y_mean.shape)
        print(f"현재가 오차율:{y_mean[0]:.2%}%")
        print(f"최고가 오차율:{y_mean[1]:.2%}%")
        print(f"최저가 오차율:{y_mean[2]:.2%}%")
        err_dict = {"cur": y_mean[0], "high": y_mean[1], "low": y_mean[2]}
        # 모델 저장
        print(model)
        with open(r"{}/coin_config/{}_err_rate".format(COIN_PATH, coin_name.lower()), "wb") as fp:
            pickle.dump(err_dict, fp)
        model.save(r"{}/coin_config/{}_lstm_model.keras".format(COIN_PATH, coin_name.lower()))

# In[ ]:


# In[ ]:



