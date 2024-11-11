import joblib
from pre_processing import data_preprocessing

# 모델과 벡터라이저 로드
nb_model = joblib.load('naive_bayes_model.joblib')
vectorizer = joblib.load('count_vectorizer.joblib')  # 벡터라이저도 저장했다고 가정

# 새로운 이메일 제목
new_email = """
중요 메일메일 제목SK텔레콤 T아이디 약관 동의 내용 정기 안내새 창으로 메일 보기
 

"""

# 형태소 분석 적용
tokenized_email = data_preprocessing.tokenize_kiwi(new_email)

# 토큰화된 텍스트를 문자열로 변환
processed_email = ' '.join(tokenized_email)

# 벡터화
email_vectorized = vectorizer.transform([processed_email])

# 예측
prediction = nb_model.predict(email_vectorized)

# 결과 출력
result = "스팸" if prediction[0] == 1 else "햄"
print(f"이 이메일은 {result}으로 분류됩니다.")

# 확률 출력 (선택사항)
probabilities = nb_model.predict_proba(email_vectorized)
spam_probability = probabilities[0][1]
print(f"스팸일 확률: {spam_probability:.2f}")




