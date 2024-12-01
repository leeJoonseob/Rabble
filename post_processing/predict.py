import joblib
import os
import numpy as np
from post_processing.classifier import spam_ham_suspicions
from pre_processing import data_preprocessing as dp
import keras


def predict_email_deep(title="", content=""):
    # 현재 스크립트의 디렉토리 경로
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # 모델 파일의 경로
    model_path = os.path.join(current_dir, '..', 'models', 'model_LSTM_MA.h5')

    # 모델 로드
    model = keras.models.load_model(model_path)

    #merge title and content
    test_text = title + " " + content

    # 형태소 분석 적용
    test_text = dp.tokenize_kiwi(test_text)
    
    #padding
    encoder = dp.TextEncoder()
    encoder.fit([test_text])
    text = encoder.encode([test_text])

    #predict
    final_pred = model.predict(text)[0][0]
    print(final_pred)

    return spam_ham_suspicions(final_pred)    


def predict_email(title="", content=""):
    # 현재 스크립트(app.py) 경로
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # models 디렉토리의 상대 경로 설정
    tfidf_path = os.path.join(current_dir, '..', 'models', 'tfidf_model.pkl')
    knn_path = os.path.join(current_dir, '..', 'models', 'knn_model.pkl')
    nb_model_path = os.path.join(current_dir, '..', 'models', 'nb_model.pkl')
    

    # 저장된 모델 로드
    tfidf = joblib.load(tfidf_path)
    knn = joblib.load(knn_path)
    nb_model = joblib.load(nb_model_path)

    test_text = title + " " + content
    text_tfidf = tfidf.transform([test_text])


    # 확률 예측
    knn_prob = knn.predict_proba(text_tfidf)[0]
    nb_prob = nb_model.predict_proba(text_tfidf)[0]
    
    # 첫 번째 확률만 사용 (정상 메일일 확률)
    knn_normal_prob = knn_prob[0]
    nb_normal_prob = nb_prob[0]
    
    # 평균 확률 계산
    avg_normal_prob = (knn_normal_prob + nb_normal_prob) / 2
    print(avg_normal_prob)
    return spam_ham_suspicions(avg_normal_prob)

