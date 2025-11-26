import numpy as np
from ai_service.Crypto_Coin_Predict import y_predict, create_predict, load_rnnmodel
from ai_service.Crypto_Coin_Train import get_data, extract_data
import os
import pickle

COIN_PATH = os.path.join(os.path.dirname(__file__), "coin_config")

print(COIN_PATH)

# BTC, ETH, XRP
# 코인 파일 prefix btc_,eth_,xrp_
service_coins = ["BTC", "ETH", "XRP"]
btc_lstm_model = load_rnnmodel(f"{COIN_PATH}/btc_lstm_model.keras")
eth_lstm_model = load_rnnmodel(r"{}\eth_lstm_model.keras".format(COIN_PATH))
xrp_lstm_model = load_rnnmodel(r"{}\xrp_lstm_model.keras".format(COIN_PATH))
btc_err_rate = ""
eth_err_rate = ""
xrp_err_rate = ""
btc_graph = ""
eth_graph = ""
xrp_graph = ""


def init_server():
    global btc_err_rate
    global eth_err_rate
    global xrp_err_rate
    global btc_graph
    global eth_graph
    global xrp_graph

    with open(f"{COIN_PATH}/btc_err_rate", "rb") as fp:
        btc_err_rate = pickle.load(fp)
    with open(f"{COIN_PATH}/eth_err_rate", "rb") as fp:
        eth_err_rate = pickle.load(fp)
    with open(f"{COIN_PATH}/xrp_err_rate", "rb") as fp:
        xrp_err_rate = pickle.load(fp)
    btc_graph = ["chart/btc_mse_mae.png", "chart/btc_scatter.png"]
    eth_graph = ["chart/eth_mse_mae.png", "chart/eth_scatter.png"]
    xrp_graph = ["chart/xrp_mse_mae.png", "chart/xrp_scatter.png"]


init_server()


def input_request(coin_name="BTC", time_gap=1):
    if not coin_name in service_coins:
        return f"{coin_name} 은 아직 서비스 하지 않는 종류입니다."
    global bct_lstm_model
    global eth_lstm_model
    global xrp_lstm_model
    global bct_err_rate
    global eth_err_rate
    global xrp_err_rate
    raw_data = get_data(f"https://api.bithumb.com/v1/candles/days?market=KRW-{coin_name}&count=200")
    preproc_data = extract_data(raw_data, False, coin_name=coin_name)
    x_pred = create_predict(preproc_data)
    if coin_name == "BTC":
        lstm_model = btc_lstm_model
    elif coin_name == "ETH":
        lstm_model = eth_lstm_model
    else:
        lstm_model = xrp_lstm_model
    y_predarr = y_predict(lstm_model, np.array([x_pred]), time_gap, coin_name)
    if coin_name == "BTC":
        grap_url = btc_graph
        err_rate = btc_err_rate
    elif coin_name == "ETH":
        grap_url = eth_graph
        err_rate = eth_err_rate
    elif coin_name == "XRP":
        grap_url = xrp_graph
        err_rate = xrp_err_rate
    print((y_predarr))
    # y_predarr = [f"{d):.4f}" for d in y_predarr]
    return {"ypred": y_predarr, "graph": grap_url, "err_rate": err_rate}


if __name__ == "__main__":
    coin_name = "BTC"
    time_gap = 5
    print(input_request(coin_name, time_gap))

