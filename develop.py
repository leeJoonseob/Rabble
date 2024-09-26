import pandas as pd
import re
import joblib
# 정상 URL 데이터 로드
benign_urls = pd.read_csv('benign_urls.csv', header=None, names=['id', 'url'])

# 필요한 열만 선택
benign_urls = benign_urls[['url']]

# 레이블 추가
benign_urls['label'] = 0

# 악성 URL 데이터 로드
malicious_urls = pd.read_csv(
    'malicious_urls.csv',
    skiprows=9,  # 상단의 주석 및 헤더를 건너뜁니다
    header=None,
    names=['id', 'dateadded', 'url', 'url_status', 'last_online', 'threat', 'tags', 'urlhaus_link', 'reporter'],
    sep=',',
    quotechar='"',
    on_bad_lines='skip',  # 문제가 있는 행을 건너뜁니다
    engine='python'
)

# 필요한 열만 선택
malicious_urls = malicious_urls[['url']]

# 레이블 추가
malicious_urls['label'] = 1

# 데이터 병합
data = pd.concat([malicious_urls, benign_urls], ignore_index=True)

# 결측치 및 빈 값 제거
data.dropna(subset=['url'], inplace=True)
data = data[data['url'].str.strip() != '']

# 데이터 타입 변환 (문자열로)
data['url'] = data['url'].astype(str)

def extract_features(url):
    features = {}
    features['url_length'] = len(url)
    features['num_digits'] = sum(c.isdigit() for c in url)
    features['num_special_chars'] = len(re.findall(r'[^\w]', url))
    # 추가적인 특징 추출 가능
    return features

# 특징 추출 적용
features = data['url'].apply(extract_features)
features_df = pd.DataFrame(list(features))

from sklearn.model_selection import train_test_split

X = features_df
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y  # 데이터 불균형을 고려하여 stratify 사용
)

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

from sklearn.metrics import accuracy_score

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'모델 정확도: {accuracy * 100:.2f}%')

# 벡터라이저 저장
joblib.dump(model, 'model.pkl')
model = joblib.load('model.pkl')
