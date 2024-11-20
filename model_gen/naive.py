import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from pre_processing import data_preprocessing

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("import 완료")

# 데이터 로드
spam_ham = pd.read_csv("Rabble/datasets/combined_data_under.csv")
logger.info("데이터 로드 완료")

# 데이터 타입 변경
spam_ham['메일종류'] = spam_ham['메일종류'].map({'햄': 0, '스팸': 1})
logger.info("데이터 타입 변경 완료")

print(spam_ham['메일종류'].value_counts())

# 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(spam_ham['메일제목'], spam_ham['메일종류'], test_size=0.2, random_state=42)
logger.info("데이터 분리 완료")

# 형태소 분석 적용
X_train_tokenized = X_train.apply(data_preprocessing.tokenize_kiwi)
X_test_tokenized = X_test.apply(data_preprocessing.tokenize_kiwi)
logger.info("형태소 분석 완료")

# 토큰화된 텍스트를 문자열로 변환
X_train_tokenized = X_train_tokenized.apply(lambda x: ' '.join(x))
X_test_tokenized = X_test_tokenized.apply(lambda x: ' '.join(x))

# CountVectorizer를 사용하여 텍스트를 숫자 벡터로 변환
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train_tokenized)
X_test_vectorized = vectorizer.transform(X_test_tokenized)
logger.info("텍스트 벡터화 완료")

# 나이브 베이즈 모델 생성 및 학습
nb_model = MultinomialNB()
nb_model.fit(X_train_vectorized, y_train)
logger.info("모델 학습 완료")

# 예측
y_pred = nb_model.predict(X_test_vectorized)

# 모델 평가
accuracy = accuracy_score(y_test, y_pred)
logger.info(f"테스트 정확도: {accuracy:.4f}")

# 분류 보고서 출력
print("\n분류 보고서:")
print(classification_report(y_test, y_pred, target_names=['햄', '스팸']))

# 모델 저장 (선택사항)
import joblib
joblib.dump(nb_model, 'naive_bayes_model.joblib')
joblib.dump(vectorizer, 'count_vectorizer.joblib')
logger.info("모델 저장 완료")