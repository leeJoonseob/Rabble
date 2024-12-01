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