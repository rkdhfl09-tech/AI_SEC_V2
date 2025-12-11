from konlpy.tag import Okt #한글 형태 분류기
import urllib.request
import pandas as pd
from tqdm import tqdm
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import Dense,Dropout
import matplotlib.pyplot as plt
#환경값
MAX_LEN = 140
MAX_TOKENS= 0
tv=None
#데이터 수집
def get_naver_nlp_data():
    urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings_train.txt",
                               filename="ratings_train.txt")
    urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings_test.txt",
                               filename="ratings_test.txt")
    print('1. 데이터 수신')
#pandas를 이용하여 데이터 로딩
def load_data():
    pd_train_data = pd.read_table('ratings_train.txt')
    pd_test_data = pd.read_table('ratings_test.txt')
    print('2-1. 훈련용 리뷰 갯수 :', len(pd_train_data))
    print('2-2. 테스트 리뷰 갯수 :', len(pd_test_data))
    print('2-3. 테이터 확인 ===== ')
    print(pd_train_data[:2])
    return pd_train_data,pd_test_data
def preprocess_data(pd_train_data,pd_test_data):
    print("데이터 유효성 확인")
    # 한글 이외 데이터 제거
    reg_han = r"[^\sㄱ-ㅎ가-힣]"
    print("한글외 삭제 실행전:", pd_train_data["document"][0])
    pd_train_data.replace(to_replace=reg_han, regex=True, inplace=True, value="")
    pd_test_data.replace(to_replace=reg_han, regex=True, inplace=True, value="")
    print("한글외 삭제 실행후:", pd_train_data["document"][0])

    train_avalid_cnt = pd_train_data["document"].isna().sum()
    test_avalid_cnt = pd_test_data["document"].isna().sum()
    print("훈련용 전체 데이터수:",len(pd_train_data))
    print("테스트 전체 데이터수:",len(pd_test_data))
    print("유효하지 않는 훈련용데이터 수량:",train_avalid_cnt)
    print("유효하지 않는 테스트데이터 수량:",test_avalid_cnt)
    # 유효하지 않는 데이터 행단위 삭제
    pd_train_data.dropna(axis=0, subset="document", inplace=True)
    pd_test_data.dropna(axis=0, subset="document", inplace=True)
    print("제거후 훈련용 전체 데이터수:", len(pd_train_data))
    print("제거후테스트 전체 데이터수:", len(pd_test_data))
    #중복데이터 확인
    print("중복된 훈련용데이터 수량:",
          len(pd_train_data)-pd_train_data["document"].nunique())  # 중복리뷰 3813
    print("중복된 테스트데이터 수량",
          len(pd_test_data)-pd_test_data["document"].nunique())  # 중복리뷰 840
    #중복데이터 제거
    pd_train_data.drop_duplicates(subset="document", inplace=True)
    pd_test_data.drop_duplicates(subset="document", inplace=True)
    print("중복된 훈련용데이터 수량:",
          len(pd_train_data) - pd_train_data["document"].nunique())  # 중복리뷰 3813
    print("중복된 테스트데이터 수량",
          len(pd_test_data) - pd_test_data["document"].nunique())  # 중복리뷰 840

    stopword = ["에서", "은", "는", "이", "가", "이다", "하다", "들", "좀", "걍", "도", "요",
                "흠", "에게", "나다", "데", "있다", "해도", "에", "의", "을", "를", "다", "한",
                "것", "내", "그", "나"]
    # 나중에 단어 출현 횟수에 따라 의미없는 단어는 추가하여 다시 제거를 하는게 좋다.
    print("한글 형태소 분리 실행")
    okt = Okt()
    x_train = []
    for doc in tqdm(pd_train_data["document"]):
        token_word = okt.morphs(doc, stem=True)  # 리턴값 : 단어 리스트
        x_train.append(" ".join([w for w in token_word if not w in stopword]))
    x_train = np.array(x_train)
    x_test = []
    for doc in tqdm(pd_test_data["document"]):
        token_word = okt.morphs(doc, stem=True)  # 리턴값 : 단어 리스트
        x_test.append(" ".join([w for w in token_word if not w in stopword]))
    x_test = np.array(x_test)
    y_train = pd_train_data["label"].to_numpy()
    y_test = pd_test_data["label"].to_numpy()
    return (x_train,y_train),(x_test,y_test)
def preprocess_empty_remove(x_data,y_data):#최종 빈 배열 삭제
    # 빈데이터 갯수 파악 및 인덱스 수집
    print("빈데이터 수량:", (np.array([len(d) for d in x_data]) <= 0).sum())
    mask_index = (np.where((np.array([len(d) for d in x_data]) > 0)))
    x_data = x_data[mask_index]
    y_data = y_data[mask_index]
    print("데이터 검증완료" if len(x_data)==len(x_data)
          else "데이터와 정답 불일치 오류" )
    return x_data,y_data
def get_max_token(x_data):
    freq_word = {}
    for p in x_train:
        # 아 더빙 진짜 짜증나다 목소리
        w_list = p.split(" ")
        # [아,더빙,진짜,짜증나다,목소리]
        for w in w_list:
            if w in freq_word:
                freq_word[w] += 1
            else:
                freq_word[w] = 1
    reverse_freq_word = {v: k for k, v in freq_word.items()}
    print(sorted(reverse_freq_word.items())[:5])
    # 10글자 이상 단어의 수량
    global MAX_TOKENS
    MAX_TOKENS = len([k for k in reverse_freq_word if k > 10])
    print("사전크기설정값:",MAX_TOKENS)
def create_vocab(x_data):
    global tv;global MAX_TOKENS
    print("MAX_TOKENS",MAX_TOKENS)
    print("xdatashape", x_data.shape)
    MAX_TOKENS +=2 #['', '[UNK]', np.str_('영화'),
    tv = tf.keras.layers.TextVectorization(
        max_tokens=MAX_TOKENS,  # 사전크기
        output_mode='int',
        pad_to_max_tokens=True,
        output_sequence_length=MAX_LEN,  # 전체문장의 길이, 자동패딩과 스피릿
    )
    tv.adapt(x_data)#사전 만들기
    print("생성된사전:",tv.get_vocabulary()[:5])
    # ['', '[UNK]', np.str_('영화'), np.str_('보다'),
    # '' +1  [UNK] +1 단어사전의 크기에 총 +2 가산합니다.
    print("사전크기:",len(tv.get_vocabulary()))
    # 어휘 사전 및 환경 저장
    vocab = tv.get_vocabulary()
    print(vocab[:5])
    save_target = {"vocab": vocab, "max_len": MAX_LEN,
                   "max_tokens": MAX_TOKENS}
    import pickle
    with open('./config/config', 'wb') as fp:
        pickle.dump(save_target, fp)
def convert_sparse(x_data):#정수로 변환
    if tv :
        x_data = tv(x_data)
        print("정수로 변환 완료:",x_data[:1])
    return x_data
def creat_model(max_tokens=MAX_TOKENS,max_len=MAX_LEN):
    print("0000000000000")
    print(max_tokens)
    print(max_len)
    emb = tf.keras.layers.Embedding(
        max_tokens + 1,
        16,
        mask_zero=True,name="embedding_3")
    adam = tf.keras.optimizers.Adam(
        learning_rate=0.0006)
    lstm_1 = tf.keras.layers.LSTM(
        16,
        return_sequences=True,name="lstm_4")
    lstm_3 = tf.keras.layers.LSTM(
        8,
        return_sequences=False,name="lstm_5")
    model = Sequential(name="sequential_2")
    model.add(Input((max_len,)))
    model.add(emb)
    model.add(lstm_1)
    model.add(lstm_3)
    model.add(Dense(32, activation="relu",name="dense_6"))
    model.add(Dropout(0.5,name="drop_4"))
    model.add(Dense(16, activation="relu",name="dense_7"))
    model.add(Dropout(0.5,name="dropout_5"))
    model.add(Dense(1, activation="sigmoid",name="dense_8"))
    model.compile(loss="binary_crossentropy", optimizer=adam, metrics=["acc"])
    return model
def fit_train(model,x_train,y_train,epoch):
    return model.fit(x_train,y_train,validation_split=0.2,
                     epochs=epoch,batch_size=len(x_train)//100)
def result_graph(history):
    plt.plot(history["loss"], label="TRAIN_LOSS")
    plt.plot(history["val_loss"], label="VALID_LOSS")
    plt.legend()
    plt.title("LOSSES")
    plt.show()
    plt.plot(history["acc"], label="TRAIN_ACC")
    plt.plot(history["val_acc"], label="VALID_LOSS")
    plt.legend()
    plt.title("ACCURACY")
    plt.show()
def print_accuracy(model,x_test,y_test):
    loss, acc = model.evaluate(x_test, y_test)
    return f"{acc*100:.2f}"
def config_save(model):
    model.save('./config/nlp_model.keras')
def upgrade_model(model,x_newdata,y_newdata):
    #모델 불러오기(가중치 데이터 불러오기)
    #사전환경 파일 불러오기
    #새로운 데이터 훈련
    #모델 다시 저장
    pass
if __name__=="__main__":
    print("해당 모듈 실행")
    #get_naver_nlp_data() # 데이터 다운로드
    pd_train_data,pd_test_data=load_data()
    (x_train,y_train),(x_test,y_test)=preprocess_data(
        pd_train_data,pd_test_data)
    print(x_train.shape)
    print(x_test.shape)
    print(y_train.shape)
    print(y_test.shape)
    x_train,y_train=preprocess_empty_remove(x_train, y_train)
    x_test, y_test = preprocess_empty_remove(x_test, y_test)
    get_max_token(x_train)
    create_vocab(x_train)#사전 생성하기
    x_train=convert_sparse(x_train)#정수변환
    model = creat_model(MAX_TOKENS,MAX_LEN) #모델 생성
    print("xxxxxxxxxxx")
    print(x_train.shape)
    print(x_train[0])
    print(y_train.shape)
    history = fit_train(model,x_train,y_train,37)# 모델 훈련
    config_save(model)
    result_graph(history.history)
    print(print_accuracy(model,x_test,y_test))#정확률 출력값




