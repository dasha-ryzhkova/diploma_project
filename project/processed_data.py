import pandas as pd
from constants import *

# завантаження обробленого датафрейму з інформацією про книги
book_train = pd.read_csv(book_train_path)
# завантаження обробленого датафрейму з інформацією про відгуки
review_train = pd.read_csv(review_train_path)

