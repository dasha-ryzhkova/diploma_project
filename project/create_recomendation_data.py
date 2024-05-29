import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from processed_data import *
from constants import *

# cстворення датафейму для рекомендацій на основі вмісту книжок
# content_based_data = book_train[['book_id', 'description']]
# content_based_data = content_based_data.set_index('book_id')


# review_train = pd.read_csv('review_info_train.csv')
# book_trian = pd.read_csv('book_info_train.csv')

# створення матриць взаємодії користувачів з елементами
interaction = review_train[['book_id', 'user_id', 'sentiment_score']]
interaction['book_id'] = interaction['book_id'].astype(np.int64)
interaction['user_id'] = interaction['user_id'].astype(np.int64)
interaction['sentiment_score'] = interaction['sentiment_score'].astype(np.float16)

# створення рейтингової матриці
rating = pd.pivot_table(interaction, index='user_id', columns='book_id',  values='sentiment_score') # values='sentiment_score', fill_value = 0)
rating = rating.fillna(0)
rating.to_csv(rating_path, index=True)
print('\nRATING MATRIX CREATED\n')
print(rating)

# створення матриці подібностей користувачів
users_sim = cosine_similarity(rating)
idx = list(rating.index)
print('\nRATING INDESES\n')
print(idx)

users_sim = pd.DataFrame(users_sim, index=idx, columns=idx)
users_sim.index.name = 'user_id'
users_sim.to_csv(user_sim_path, index=True)





