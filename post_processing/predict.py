import joblib
import os

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

    # 결과 비교 및 출력
    if knn_pred == nb_pred:
        # 두 모델의 결과가 동일할 경우, 하나의 결과만 출력
        result = "스팸" if knn_pred == 1 else "햄"
        return result  
    else:
        # 두 모델의 결과가 다를 경우, "스팸 의심 메일"로 출력
        return "스팸 의심 메일"
