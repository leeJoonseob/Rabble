import tensorflow.keras as keras
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


model = keras.models.load_model('model_LSTM.h5')


#prepare data #example for spam message
dataFrame = pd.DataFrame(["엘지유플러스 재약정안내문 [Web발신](광고)우수고객님 감사합니다.안녕하십니까~엘지 유플러스고객 행복지점입니다. ☏ 상담번호 ☏☞070)4499-3293"])

#data to text
texts = dataFrame[0].astype(str).tolist()

#tokenizing the data
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)#fit on texts
sequences = tokenizer.texts_to_sequences(texts)

#padding
max_len = 100  # 모델 학습 시 사용했던 max_len과 동일한 값으로 설정
sequences_padded = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')

predictions = model.predict(sequences_padded)
print(predictions)


spam_class = ['햄', '스팸']

predicted_class = predictions.argmax(axis=-1)
print(spam_class[predicted_class[0]])

