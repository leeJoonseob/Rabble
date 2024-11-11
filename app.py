from flask import Flask, request, jsonify
from pre_processing import data_preprocessing
from post_processing import classifier
import joblib
from flask_cors import CORS

app = Flask(__name__)

# CORS 설정: 모든 도메인에서의 요청을 허용
CORS(app, origins=["http://localhost:3000"])

# 모델 로드
nb_model = joblib.load('naive_bayes_model.joblib')
vectorizer = joblib.load('count_vectorizer.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    title = data['title']
    content = data['content']
    
    # 제목과 내용 결합
    text = title + " " + content
    
    # 형태소 분석 적용
    tokenized_email = data_preprocessing.tokenize_kiwi(text)

    # 토큰화된 텍스트를 문자열로 변환
    processed_email = ' '.join(tokenized_email)

    # 벡터화
    email_vectorized = vectorizer.transform([processed_email])

    # 예측
    prediction = nb_model.predict(email_vectorized)

    # 결과 출력
    result = "스팸" if prediction[0] == 1 else "햄"
    
    return jsonify({'result': result})

if __name__ == '__main__':
    app.run(debug=True, port=5000)  