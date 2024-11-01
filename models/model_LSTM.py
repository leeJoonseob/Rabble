from pre_processing import data_preprocessing
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding,Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

import numpy as np
import matplotlib.pyplot as plt

#load data
spam_ham  = pd.read_csv("Rabble/datasets/combined_data.csv")

#change data type to numeric
spam_ham['메일종류'] = spam_ham['메일종류'].map({'햄':0, '스팸':1})



#split data
X_train,X_test,y_train,y_test = train_test_split(spam_ham['메일제목'], spam_ham['메일종류'], test_size=0.2, random_state=42)

#tokenizing
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)#seperate and indexing words


X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)

#padding - make all data to same length
max_len = max(len(i) for i in X_train)
X_train = sequence.pad_sequences(X_train, maxlen=max_len, padding='post', truncating='post')
X_test = sequence.pad_sequences(X_test, maxlen=max_len, padding='post', truncating='post')

#one-hot encoding
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Vocabulary size
vocab_size = len(tokenizer.word_index) + 1  # +1 for padding token
#to check the size of vocabulary to use in embedding layer input_dim
print(f"Vocabulary Size: {vocab_size}")

#sturuucture of model
model = Sequential()
model.add(Embedding(input_dim = vocab_size,output_dim = 100, input_length=max_len))
model.add(LSTM(100, activation='tanh',return_sequences=True))
model.add(Dropout(0.5))
model.add(LSTM(50, activation='tanh'))
model.add(Dense(2, activation='softmax'))


#compile
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])#왜 categorical_crossentropy인가?

#early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5,restore_best_weights=True)

#train
history = model.fit(X_train, y_train, batch_size=20, epochs=200, 
                    validation_data=(X_test, y_test), callbacks=[early_stopping])

#evaluation
print("\n Test Accuracy: %.4f" % (model.evaluate(X_test, y_test)[1]))

y_vloss = history.history['val_loss']
y_loss = history.history['loss']

model.save('model_LSTM.h5')