import tensorflow.keras as keras
from pre_processing import converter
from post_processing import classifier

model = keras.models.load_model('model_LSTM.h5')
text = "엘지유플러스 재약정안내문 [Web발신](광고)우수고객님 감사합니다.안녕하십니까~엘지 유플러스고객 행복지점입니다. ☏ 상담번호 ☏☞070)4499-3293"

sequences_padded = converter.encode_data([text])


predictions = model.predict(sequences_padded)
print(predictions)

spam_class = ['햄', '스팸']

print(classifier.spam_ham_suspicions(predictions))



