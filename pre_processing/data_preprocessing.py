import pandas as pd
import re
from kiwipiepy import Kiwi
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence

class KiwiSingleton:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = Kiwi()
        return cls._instance

def combine_data():
    # 학습 데이터 로드
    google_message = pd.read_csv("Rabble/datasets/messages.csv")
    spam_message = pd.read_csv("Rabble/datasets/spam_virus.csv")
    spam_msg = pd.read_csv("Rabble/datasets/spam_dataset.csv")
    
    # 메일 종류 추가
    google_message['메일종류'] = '햄'
    spam_message['메일종류'] = '스팸'
    spam_msg['메일종류'] = '스팸'

    # 필요한 열 선택
    final_google_df = google_message[['메일종류', '메일제목']]
    final_spam_df = spam_message[['메일종류', '메일제목']]
    final_spam_msg_df = spam_msg[['메일종류', '메일제목']]


    # Ham 데이터 수
    ham_count = final_google_df.shape[0]

    # Spam 데이터 언더샘플링
    final_spam_df = final_spam_df.sample(n=ham_count, random_state=42)

    # 데이터 결합
    combined_df = pd.concat([final_google_df, final_spam_df, final_spam_msg_df], axis=0, ignore_index=True)

    combined_df.to_csv('/Users/PROTEIN/Desktop/Coll_Third/spam_finder/Rabble/datasets/combined_data_under.csv', index=False, encoding='utf-8')

def tokenize_kiwi(text):
    kiwi = KiwiSingleton.get_instance()
    analyzed = kiwi.analyze(text)
    return [token.form for token in analyzed[0][0]]

class TextEncoder:
    def __init__(self):
        self.tokenizer = None
        self.max_len = None
        self.vocab_size = None

    def fit(self, text):
        self.tokenizer = Tokenizer()
        self.tokenizer.fit_on_texts(text)
        self.vocab_size = len(self.tokenizer.word_index) + 1
        self.max_len = max(len(seq) for seq in self.tokenizer.texts_to_sequences(text))

    def encode(self, text):
        if not self.tokenizer:
            raise ValueError("Tokenizer not fitted. Call fit() first.")
        sequences = self.tokenizer.texts_to_sequences(text)
        return sequence.pad_sequences(sequences, maxlen=self.max_len, padding='post', truncating='post')

combine_data()