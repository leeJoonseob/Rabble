import joblib
import os
import numpy as np
from post_processing.classifier import spam_ham_suspicions

def predict_email(title, content):
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
    test_tfidf = tfidf.transform([test_text])

    # K-NN 예측
    knn_pred = knn.predict(test_tfidf)[0]
    nb_pred = nb_model.predict(test_tfidf)[0]

    #Soft Voting
    avg_pred = (knn_pred + nb_pred) / 2

    # 최종 예측 결정
    final_pred = np.argmax(avg_pred, axis=1)[0]

    return spam_ham_suspicions(final_pred)
    