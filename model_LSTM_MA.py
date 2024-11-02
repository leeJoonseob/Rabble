from pre_processing import data_preprocessing

import pandas as pd

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding,Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split


import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("import 완료")


#load data
spam_ham  = pd.read_csv("Rabble/datasets/combined_data.csv")
logger.info("데이터 로드 완료")

#change data type to numeric
spam_ham['메일종류'] = spam_ham['메일종류'].map({'햄':0, '스팸':1})
logger.info("데이터 타입 변경 완료")


#split data
X_train,X_test,y_train,y_test = train_test_split(spam_ham['메일제목'], spam_ham['메일종류'], test_size=0.2, random_state=42)
logger.info("데이터 분리 완료")

# 형태소 분석 적용
X_train_tokenized = X_train.apply(data_preprocessing.tokenize_kiwi)
X_test_tokenized = X_test.apply(data_preprocessing.tokenize_kiwi)
logger.info("형태소 분석 완료")

#padding - make all data to same length
max_len = max(len(i) for i in X_train)
encoder = data_preprocessing.TextEncoder()
encoder.fit(X_test_tokenized)
X_train = encoder.encode(X_train)
X_test = encoder.encode(X_test_tokenized)
logger.info("패딩 완료")

#one-hot encoding
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
logger.info("원핫인코딩 완료")

#to check the size of vocabulary to use in embedding layer input_dim
vocab_size = encoder.vocab_size

#sturuucture of model
model = Sequential()
model.add(Embedding(input_dim = vocab_size,output_dim = 100, input_length=max_len))
model.add(LSTM(100, activation='tanh',return_sequences=True))
model.add(Dropout(0.5))
model.add(LSTM(50, activation='tanh'))
model.add(Dense(2, activation='softmax'))
logger.info("모델 구조 생성")

#compile
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])#왜 categorical_crossentropy인가?
logger.info("모델 컴파일 완료")

#early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5,restore_best_weights=True)

logger.info("학습 시작")
#train
history = model.fit(X_train, y_train, batch_size=20, epochs=200, 
                    validation_data=(X_test, y_test), callbacks=[early_stopping])

#evaluation
print("\n Test Accuracy: %.4f" % (model.evaluate(X_test, y_test)[1]))

y_vloss = history.history['val_loss']
y_loss = history.history['loss']

model.save('model_LSTM_MA.h5')