from flask import Flask, request, jsonify
from post_processing.predict import predict_email
from flask_cors import CORS

app = Flask(__name__)

# CORS 설정: 모든 도메인에서의 요청을 허용
CORS(app, origins=["http://localhost:3000"])

# # 모델 로드
# nb_model = joblib.load('Rabble/models/nb_model.pkl')
# vectorizer = joblib.load('count_vectorizer.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    title = data['title']
    content = data['content']
    
    result = predict_email(title, content)
    
    return jsonify({'result': result})

if __name__ == '__main__':
    app.run(debug=True, port=5000)  