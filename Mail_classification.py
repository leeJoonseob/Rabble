from pre_processing import data_preprocessing as dp

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib
from sklearn.model_selection import train_test_split

# 데이터와 모델을 저장할 전역 변수
tfidf = None
knn = None
nb_model = None

    
def process_data(data):
    # 형태소 분석 적용
    # Morphological Analysis
    data_tokenized = data.apply(dp.tokenize_kiwi)
    

    #padding - make all data to same length
    encoder = dp.TextEncoder()
    encoder.fit(data_tokenized)
    data = encoder.encode(data_tokenized)
    
    return data
    

def combined_data():
    google_message = pd.read_csv("Rabble/datasets/messages.csv")
    spam_message = pd.read_csv("Rabble/datasets/spam_virus.csv")
    google_message['메일종류'] = '햄'
    spam_message['메일종류'] = '스팸'
    google_message['메일제목'] = google_message['메일제목'] + ' ' + google_message['메일내용']
    final_google_df = google_message[['메일종류', '메일제목']]
    final_spam_df = spam_message[['메일종류', '메일제목']]

    ham_count = final_google_df.shape[0]
    final_spam_df = final_spam_df.sample(n=ham_count, random_state=33)
    combined_df = pd.concat([final_google_df, final_spam_df], axis=0, ignore_index=True)
    return combined_df

def train_model():
    global tfidf, knn, nb_model

    combined_data_df = pd.read_csv("Rabble/datasets/combined_data_under.csv")
    X = combined_data_df['메일제목']
    y = combined_data_df['메일종류'].apply(lambda x: 1 if x == '스팸' else 0)

    X_train, X_test, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    #K-NN 모델 학습을 위한 TF-IDF 벡터화
    tfidf = TfidfVectorizer(max_features=3000)
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_val_tfidf = tfidf.transform(X_test)

    #나이브 베이즈 모델 학습을 위한 kiwi 형태소 분석
    X_train_kiwi, X_test_kiwi = process_data(X_train, X_test)

    # K-NN 모델 학습
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train_tfidf, y_train)
    joblib.dump(knn, 'knn_model.pkl')

    # 나이브 베이즈 모델 학습
    nb_model = MultinomialNB()
    nb_model.fit(X_train_kiwi, y_train)
    joblib.dump(nb_model, 'nb_model.pkl')

    # TF-IDF 모델 저장
    joblib.dump(tfidf, 'tfidf_model.pkl')

    # 모델 평가
    knn_pred = knn.predict(X_val_tfidf)
    nb_pred = nb_model.predict(X_test_kiwi)

    print('K-NN 정확도:', accuracy_score(y_val, knn_pred))
    print('K-NN 혼동 행렬:\n', confusion_matrix(y_val, knn_pred))
    
    print('나이브 베이즈 정확도:', accuracy_score(y_val, nb_pred))
    print('나이브 베이즈 혼동 행렬:\n', confusion_matrix(y_val, nb_pred))

    return knn, nb_model

# # 모델 학습
# train_model()




