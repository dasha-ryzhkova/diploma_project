import pandas as pd
import time
import os

from gensim.models import Word2Vec

from constants import *
from help_func import *
from processed_data import book_train, review_train

# очищення датафреймів від рядків з пустими текстовими полями 
book_train = book_train.dropna(subset=['description'])
review_train = review_train.dropna(subset=['text'])


w2v_train = []
# створення тренувального словника з тексту відгуків
l = list(review_train['text'])
for i in l:
    w2v_train.append(i.split(' '))
# створення тренувального словника з тексту опису книг
l = list(book_train['description'])
for i in l:
    w2v_train.append(i.split(' '))


t1 = time.time()
# ініціація иа навчання моделі Word2Vec
w2v_model = Word2Vec(w2v_train, 
                     epochs=EPOCHS_W2V,
                     vector_size=VECTOR_SIZE, 
                     window=WINDOW_W2V, 
                     min_count=MIN_COUNT_W2V, 
                     workers=WORKERS_W2V, 
                     sg=0) # CBOW

# збереження Word2Vec
w2v_model.save(w2v_path)
t2 = time.time()

# час тренування Word2Vec
s = 'Time for World2Vec training: {:.4f} minutes.\n'.format((t2 - t1) / 60)
write_time(s)
# f = open('time.txt', "w")
# f.write(s)
# f.close()
