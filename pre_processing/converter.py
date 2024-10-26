import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def encode_data(text):
    """
    return: 
    This function will encode the data to numeric values
    """

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(text)#fit on texts
    sequences = tokenizer.texts_to_sequences(text)

    max_len = 100  # 모델 학습 시 사용했던 max_len과 동일한 값으로 설정
    sequences_padded = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')

    return sequences_padded
