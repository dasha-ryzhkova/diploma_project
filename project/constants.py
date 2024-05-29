import os
# import nltk
# from nltk.corpus import stopwords

# nltk.download('punkt')
# nltk.download('stopwords')
# STOP_WORDS = set(stopwords.words('english'))




# шлях до основних даних
path = './data'
name1 = 'books_data.csv'
name2 = 'Books_rating.csv'

file1 = os.path.join(path, name1)
file2 = os.path.join(path, name2)
# шлях до файлу зі стоп-словами
stop_words_file = os.path.join(path, 'stop_words.txt')
# шлях до файлу з необробленим даними
book_train_path = os.path.join(path, 'book_info_train.csv') 
review_train_path = os.path.join(path, 'review_info_train.csv') 
# шлях до файлу з рейтинговою матрицею та матрицею подібності користувачів
rating_path = os.path.join(path, 'rating.csv') 
user_sim_path = os.path.join(path, 'users_sim.csv') 

# шлях до тестових файлів
test_path = './test'
test_file1 = os.path.join(test_path, 'test1.csv')
test_file2 = os.path.join(test_path, 'test2.csv')
test_file3 = os.path.join(test_path, 'test3.csv')

test_files = [test_file1, test_file2, test_file3]

# шлях до директорії для збереження навчених моделей
model_path = './models'
# шлях до моделі W2V
w2v_path = os.path.join(model_path, 'w2v_colab.model')
# шлях до tokenizer
tokenizer_path = os.path.join(model_path, 'tokenizer.pickle')
# шлях до моделі LSTM
sent_path = os.path.join(model_path, 'sentiment.keras')

# параметри для моделі W2V
EPOCHS_W2V=20       # кількість епох навчання
VECTOR_SIZE=100     # довжина вектору пркдставлення / розмірність простору
WINDOW_W2V=5        # вікно контексту
MIN_COUNT_W2V=5     # мінімальна кількість слів для додавання в словник
WORKERS_W2V=4       # робочі потоки для навчання

# параметри для моделі LSTM
VOCAB_LEN=300000    # довжина словника
DATA_LEN=100        # довжина вектору пркдставлення / розмірність простору
EPOCHS_SENT=70      # кількість епох навчання


# збереження стоп-слів з файлу у список
with open(stop_words_file) as f:
    STOP_WORDS = f.read().splitlines()

#print(STOP_WORDS)
# with open(stop_words_file, 'a') as f:
#     for word in STOP_WORDS:
#         f.write(f'{word}\n')
