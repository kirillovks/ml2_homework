import json
import ast
import pandas as pd
import re
from nltk.corpus import stopwords
from tqdm import tqdm
from nltk.stem import WordNetLemmatizer
import swifter
import nltk
import pickle
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))
#self.stop_words.extend(['unused word1', 'unused word2']) - слова которые добавить в стопвордс - по результатам EDA
lemmatizer = WordNetLemmatizer()

def get_mean_w2v_vector(sentence):
    model = pickle.load(open('../models/word2vecmodel.pkl', 'rb'))
    Sum = 0
    Count = 0
    
    try:
        words = str(sentence).split()
    except TypeError:
        words = []
        
    for w in words:
        if w in model.wv:
            Sum += model.wv[w]
            Count += 1
            
    if Count == 0:
        return np.zeros(model.vector_size)
        
    return Sum / Count


def clean_text(phrase):
    cleared_text = re.sub(r'[^a-zA-Z\s]', ' ', phrase)  # Чистим текст
    words = cleared_text.lower().split()  # Токенизация и приведение к нижнему регистру
    filtered_words = [word for word in words if word not in stop_words]  # Убираем стоп-слова
    lemmatized_words = [lemmatizer.lemmatize(word) for word in filtered_words]  # Лемматизация
    return ' '.join(lemmatized_words).strip()


def vectorize(text, vectors_dim = 300):
    HIDDEN = vectors_dim
    text_vectors = text.map(get_mean_w2v_vector)
    text = pd.DataFrame(text_vectors.tolist(), index=text.index)
    return text