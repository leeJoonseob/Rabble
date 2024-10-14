import matplotlib.font_manager as fm
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib as mpl

# 한글 폰트 설정
#plt.rcParams['font.family'] = 'AppleGothic'  # MacOS인 경우
# plt.rcParams['font.family'] = 'Malgun Gothic'  # Windows인 경우
# plt.rcParams['font.family'] = 'NanumGothic'  # Linux 또는 다른 폰트를 사용하는 경우
#print(plt.rcParams['font.family']) 

# 데이터 불러오기
spam_virus = pd.read_csv("Rabble/datasets/spam_virus.csv")
print(spam_virus.head())


spam_virus['수신일자'] = pd.to_datetime(spam_virus['수신일자'])
spam_virus['수신일자'] = spam_virus['수신일자'].astype(int) / 10**9

spam_virus['수신시간'] = spam_virus['수신시간'].apply(lambda x: f"{x}:00" if len(x.split(':')) == 2 else x)
spam_virus['수신시간'] = pd.to_timedelta(spam_virus['수신시간']).dt.total_seconds()

mapping = {'스팸': 1, '바이러스': 2}
spam_virus['메일종류'] = spam_virus['메일종류'].map(mapping)

# TF-IDF 벡터화 수행
vectorizer = TfidfVectorizer(max_features=100)  # 최대 100개의 특성을 사용
tfidf_matrix = vectorizer.fit_transform(spam_virus['메일제목'])

# TF-IDF 결과를 데이터프레임으로 변환
tfidf_df = pd.DataFrame(
    tfidf_matrix.toarray(),
    columns=vectorizer.get_feature_names_out(),
    index=spam_virus.index
)

# '메일제목' 열을 삭제하고 TF-IDF 데이터로 대체
spam_virus = spam_virus.drop('메일제목', axis=1)
spam_virus = pd.concat([spam_virus, tfidf_df], axis=1)

spam_virus['첨부'] = spam_virus['첨부'].map(lambda x: 1 if x != '없음' else 0)

# 상관계수 히트맵
mpl.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 12
sns.set(rc={'figure.figsize':(15,15)})
correlation_matrix = spam_virus.corr().round(2)
sns.heatmap(data=correlation_matrix, annot=True)
plt.show()


tfidf_columns = spam_virus.columns[:]
# 각 단어(열)의 TF-IDF 값 합계 계산
word_importance = spam_virus[tfidf_columns].sum().sort_values(ascending=False)

# 상위 20개 단어 출력
print(word_importance.head(60))

spam_virus = spam_virus[["수신일자","수신시간","메일종류","첨부","광고","회원","가입을","주식","posa","предложение","payment","한번","드립니다","해외선물","선물","선착순","보장","해드릴께요","태국","혼자","보실래요","밤에","신나게","마세요"]]