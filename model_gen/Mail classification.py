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

def combined_data():
    google_message = pd.read_csv("Rabble/datasets/messages.csv")
    spam_message = pd.read_csv("Rabble/datasets/spam_virus.csv")
    google_message['메일종류'] = '햄'
    spam_message['메일종류'] = '스팸'
    final_google_df = google_message[['메일종류', '메일제목']]
    final_spam_df = spam_message[['메일종류', '메일제목']]

    ham_count = final_google_df.shape[0]
    final_spam_df = final_spam_df.sample(n=ham_count, random_state=42)
    combined_df = pd.concat([final_google_df, final_spam_df], axis=0, ignore_index=True)
    return combined_df

def train_model():
    global tfidf, knn, nb_model

    combined_data = pd.read_csv("Rabble/datasets/combined_data_under.csv")
    X = combined_data['메일제목']
    y = combined_data['메일종류'].apply(lambda x: 1 if x == '스팸' else 0)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

    tfidf = TfidfVectorizer(max_features=3000)
    X_train = tfidf.fit_transform(X_train)
    X_val = tfidf.transform(X_val)

    # K-NN 모델 학습
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    joblib.dump(knn, 'knn_model.pkl')
    print(knn.score(X_val, y_val))
    # 나이브 베이즈 모델 학습
    nb_model = MultinomialNB()
    nb_model.fit(X_train, y_train)
    joblib.dump(nb_model, 'nb_model.pkl')
    print(nb_model.score(X_val, y_val))
    # TF-IDF 모델 저장
    joblib.dump(tfidf, 'tfidf_model.pkl')

    return knn, nb_model

# 모델 학습
train_model()




