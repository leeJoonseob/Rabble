from pre_processing import data_preprocessing as dp

import pandas as pd

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Embedding,GlobalAveragePooling1D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from tensorflow.keras.regularizers import l1, l2, l1_l2
from tensorflow.keras.optimizers import Adam


import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("import 완료")


#load data
spam_ham  = pd.read_csv("Rabble/datasets/combined_data_under.csv")
logger.info("데이터 로드 완료")

#change data type to numeric
spam_ham['메일종류'] = spam_ham['메일종류'].map({'햄':0, '스팸':1})
logger.info("데이터 타입 변경 완료")


#split data
X_train,X_test,y_train,y_test = train_test_split(spam_ham['메일제목'], spam_ham['메일종류'], test_size=0.3, random_state=42)
logger.info("데이터 분리 완료")


# 형태소 분석 적용
# Morphological Analysis
X_train_tokenized = X_train.apply(dp.tokenize_kiwi)
X_test_tokenized = X_test.apply(dp.tokenize_kiwi)
logger.info("형태소 분석 완료")

#padding - make all data to same length
max_len = max(len(i) for i in X_train)
encoder = dp.TextEncoder()
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

def create_regularized_model(input_dim, output_dim):
    model = Sequential([
        Embedding(input_dim = vocab_size,output_dim = 64, input_length=max_len),
        GlobalAveragePooling1D(),
        # L1 정규화
        Dense(64, activation='relu', input_shape=(input_dim,),
              kernel_regularizer=l1(0.01)),
        Dropout(0.3),  # 30% 드롭아웃
        
        # L2 정규화
        Dense(32, activation='relu',
              kernel_regularizer=l2(0.01)),
        Dropout(0.3),  # 30% 드롭아웃
        
        # L1 및 L2 정규화 동시 적용
        Dense(16, activation='relu',
              kernel_regularizer=l1_l2(l1=0.01, l2=0.01)),
        Dropout(0.3),  # 30% 드롭아웃
        
        Dense(output_dim, activation='softmax')
    ])
    
    # 모델 컴파일
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

# 모델 생성 (입력 차원과 출력 차원을 적절히 조정하세요)
input_dim = vocab_size  # 예: 입력 특성의 수
output_dim = 2   # 예: 분류 클래스의 수
model = create_regularized_model(input_dim, output_dim)
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