import tensorflow as tf
import pandas as pd
from transformers import BertTokenizer, TFBertForSequenceClassification
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from sklearn.model_selection import train_test_split

# BERT 토크나이저 로드
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=False)

# 데이터 전처리 함수
def preprocess_function(texts):
    return tokenizer(texts, truncation=True, padding='max_length', max_length=128, return_tensors='tf')

# 데이터 로드 및 전처리
spam_ham = pd.read_csv("Rabble/datasets/combined_data_under.csv")
X = spam_ham['메일제목'].tolist()
y = spam_ham['메일종류'].map({'햄': 0, '스팸': 1}).values

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 데이터 토큰화
train_encodings = preprocess_function(X_train)
test_encodings = preprocess_function(X_test)

# 데이터셋 생성
train_dataset = tf.data.Dataset.from_tensor_slices((
    dict(train_encodings),
    y_train
)).shuffle(1000).batch(16)

test_dataset = tf.data.Dataset.from_tensor_slices((
    dict(test_encodings),
    y_test
)).batch(16)

# BERT 모델 로드
model = TFBertForSequenceClassification.from_pretrained('bert-base-multilingual-cased', num_labels=2)

# 모델 컴파일
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 모델 학습
history = model.fit(
    train_dataset,
    epochs=10,
    validation_data=test_dataset
)

# 모델 평가
results = model.evaluate(test_dataset)
print(f"Test Loss: {results[0]}")
print(f"Test Accuracy: {results[1]}")

# 모델 저장
model.save_pretrained('bert_spam_classifier')