from pre_processing import data_preprocessing

import pandas as pd
import tensorflow as tf
import numpy as np

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding, Dropout, Conv1D, MaxPooling1D, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from tensorflow.keras.optimizers.schedules import ExponentialDecay


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
X_train,X_test,y_train,y_test = train_test_split(spam_ham['메일제목'], spam_ham['메일종류'], test_size=0.2, random_state=42)
logger.info("데이터 분리 완료")

from nltk.corpus import wordnet

def synonym_replacement(sentence, n=1):
    words = sentence.split()
    new_words = words.copy()
    for _ in range(n):
        word = np.random.choice(words)
        synonyms = wordnet.synsets(word)
        if synonyms:
            synonym = synonyms[0].lemmas()[0].name()
            new_words = [synonym if w == word else w for w in new_words]
    return ' '.join(new_words)

def random_insertion(sentence, n=1):
    words = sentence.split()
    for _ in range(n):
        word = np.random.choice(words)
        synonyms = wordnet.synsets(word)
        if synonyms:
            synonym = synonyms[0].lemmas()[0].name()
            position = np.random.randint(0, len(words) + 1)
            words.insert(position, synonym)
    return ' '.join(words)

X_train_aug_replace = X_train.apply(lambda x: synonym_replacement(x, n=1))
X_train_aug_insert = X_train.apply(lambda x: random_insertion(x, n=1))


X_train = pd.concat([X_train, X_train_aug_replace, X_train_aug_insert], axis=0)
y_train = pd.concat([y_train, y_train, y_train], axis=0)
# 형태소 분석 적용
X_train = X_train.apply(data_preprocessing.tokenize_kiwi)
X_test = X_test.apply(data_preprocessing.tokenize_kiwi)
logger.info("형태소 분석 완료")
print(X_train)

#padding - make all data to same length
max_len = max(len(i) for i in X_train)
encoder = data_preprocessing.TextEncoder()
encoder.fit(X_test)
X_train = encoder.encode(X_train)
X_test = encoder.encode(X_test)
logger.info("패딩 완료")
print(X_train[0])
print(X_train[1])



#change data type to numpy array
X_train = np.array(X_train)
y_train = np.array(y_train)

#one-hot encoding
# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)
# logger.info("원핫인코딩 완료")

#to check the size of vocabulary to use in embedding layer input_dim
vocab_size = encoder.vocab_size

def create_model(vocab_size, max_len):
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=32, input_length=max_len),
        Dropout(0.3),
        Conv1D(16, 5, activation='relu', strides=1, kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        MaxPooling1D(pool_size=4),
        LSTM(16, kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        Dense(8, activation='relu', kernel_regularizer=l2(0.01)),
        Dropout(0.3),
        Dense(2, activation='softmax')
    ])
    return model

#learning rate scheduler
initial_learning_rate = 0.001
lr_schedule = ExponentialDecay(
    initial_learning_rate,
    decay_steps=100000,
    decay_rate=0.96,
    staircase=True)

# K-fold Cross Validation 설정
kfold = KFold(n_splits=10, shuffle=True, random_state=42)

# K-fold Cross Validation 모델 평가
fold_no = 1
acc_per_fold = []
loss_per_fold = []

for train, val in kfold.split(X_train, y_train):
    logger.info(f"Training for fold {fold_no} ...")

    #model create
    model = create_model(vocab_size, max_len)

    optimizer = Adam(learning_rate=lr_schedule)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    # 콜백 정의
    early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    check_point = tf.keras.callbacks.ModelCheckpoint(
        f"LSTM_CNN_model_fold_{fold_no}.keras",
        monitor="val_accuracy",
        save_best_only=True,
        mode='max'
    )
    

    # 모델 학습
    history = model.fit(
        X_train[train], y_train[train],
        batch_size=20,
        epochs=200,
        validation_data=(X_train[val], y_train[val]),
        callbacks=[early_stopping, check_point]
    )

    # 모델 평가
    scores = model.evaluate(X_train[val], y_train[val], verbose=0)
    logger.info(f"Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%")
    acc_per_fold.append(scores[1] * 100)
    loss_per_fold.append(scores[0])

    fold_no += 1

# 평균 점수 출력
logger.info('------------------------------------------------------------------------')
logger.info('Score per fold')
for i in range(0, len(acc_per_fold)):
    logger.info(f'> Fold {i+1} - Loss: {loss_per_fold[i]} - Accuracy: {acc_per_fold[i]}%')
logger.info('------------------------------------------------------------------------')
logger.info('Average scores for all folds:')
logger.info(f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})')
logger.info(f'> Loss: {np.mean(loss_per_fold)}')
logger.info('------------------------------------------------------------------------')

# 최종 테스트 세트에 대한 평가
final_model = tf.keras.models.load_model(f"LSTM_CNN_model_fold_{np.argmax(acc_per_fold)+1}.keras")
test_loss, test_accuracy = final_model.evaluate(X_test, y_test)
logger.info(f"Final Test Accuracy: {test_accuracy*100:.2f}%")
