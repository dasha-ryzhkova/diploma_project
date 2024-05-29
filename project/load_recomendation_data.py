import pandas as pd
import pickle
import keras 
from gensim.models import Word2Vec

from processed_data import *
from constants import *



# створення датафрейму з id книги та її описом
content_based_data = book_train[['book_id', 'description']]
content_based_data = content_based_data.set_index('book_id')
print('content_based_data loaded!!!')

# завантаження рейтингової матриці 
rating = pd.read_csv(rating_path)
rating = rating.set_index('user_id')
idx = rating.index
cols = rating.columns
# перетворення датаврейму у двомірний масив
rating_np = rating.to_numpy()
print('rating loaded!!!')

# завантаження матриці подібності користувачів
users_sim = pd.read_csv(user_sim_path)
users_sim = users_sim.set_index('user_id')
# перетворення датаврейму у двомірний масив
user_sim_np = users_sim.to_numpy()
print('users_sim loaded!!!')


# Завантаження навчених моделей
w2v_model = Word2Vec.load(w2v_path)
sent_model = keras.models.load_model(sent_path)   

with open(tokenizer_path, 'rb') as f:
    tokenizer = pickle.load(f)
emb_vec = w2v_model.wv

print('Everything loaded!!!')