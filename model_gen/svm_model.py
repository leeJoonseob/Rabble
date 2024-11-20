from pre_processing import data_preprocessing as dp

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("import 완료")

# 데이터 로드
spam_ham  = pd.read_csv("Rabble/datasets/combined_data_under.csv")
logger.info("데이터 로드 완료")

# 데이터 타입 변경
spam_ham['메일종류'] = spam_ham['메일종류'].map({'햄':0, '스팸':1})
logger.info("데이터 타입 변경 완료")

# 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(spam_ham['메일제목'], spam_ham['메일종류'], test_size=0.3, random_state=42)
logger.info("데이터 분리 완료")

# 형태소 분석 적용
# Morphological Analysis
X_train_tokenized = X_train.apply(dp.tokenize_kiwi)
X_test_tokenized = X_test.apply(dp.tokenize_kiwi)
logger.info("형태소 분석 완료")

# 토큰화된 텍스트를 문자열로 변환
X_train_joined = X_train_tokenized.apply(lambda x: ' '.join(x))
X_test_joined = X_test_tokenized.apply(lambda x: ' '.join(x))

# TF-IDF 벡터화
tfidf = TfidfVectorizer(max_features=3000)
X_train_tfidf = tfidf.fit_transform(X_train_joined)
X_test_tfidf = tfidf.transform(X_test_joined)
logger.info("TF-IDF 벡터화 완료")

# SVM 모델 생성
model = svm.SVC(kernel='linear', probability=True)
logger.info("SVM 모델 생성 완료")

logger.info("학습 시작")
# 모델 학습
model.fit(X_train_tfidf, y_train)
logger.info("학습 완료")

# 예측
y_pred = model.predict(X_test_tfidf)

# 정확도 평가
accuracy = accuracy_score(y_test, y_pred)
print("\n Test Accuracy: %.4f" % accuracy)

# 모델 저장
if accuracy > 0.8:
    logger.info("모델 저장")
    import joblib
    joblib.dump(model, 'svm_model.pkl')
    logger.info("모델 저장 완료")
else:
    logger.info("모델 저장 실패: 정확도가 너무 낮습니다.")