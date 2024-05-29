import pandas as pd
import numpy as np


import time
from nltk.corpus import stopwords


import warnings
warnings.filterwarnings(action='ignore')

from constants import *
from help_func import *

def get_test_data(df, file, n=3):
    """
    Створення файлі для тестування
    Input: df - датафрейм з коментарями; 
           file - файл, у який зберігається утворений датафрейм; 
           n - кількість зразків у тестових файлах.
    Output: df_train - датафрейм без рядків, що записані в тестовий файл
    """
    # зміна порядку рядків
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df_test = df[:n]
    df_train = df[n:]
    # обирання необхідних полів
    df_test = df_test[['book_id', 'user_id', 'text']]
    print(df_test)
    # запис у файл
    df_test.to_csv(file, index=False)
    return df_train


def prepare_dataframes(file1, file2, test_files):
    """
    Підготовка датафреймів до роботи. Заміна індексів типу string на int. 
    Повязання датафреймів через поле book_id. Зміна порядку рядків датафреймів коментрарів.
    Перейменування полів. Видалення рядків з пустими полями. Нормалізація поля sentiment_score.
    Створення поля score.
    Input: file1 - назва файл з інформацією про книги;
           file2 - назва файл з інформацією про відгуки; 
           test_files - список назв для тестових файлів.
    Output: df1 - датафрейм з інформацією про книги;
            df2 - датафрейм з інформацією про відгуки.
    """

    df1 = pd.read_csv(file1, delimiter=',')
    df2 = pd.read_csv(file2, delimiter=',')

    # виділення id книжок
    book_ids = df2['Id'].unique()
    # заміна id-string на id-int
    book_to_index = {item_id: idx for idx, item_id in enumerate(book_ids, 1)}
    df2['Id'] = df2['Id'].map(book_to_index) #replace({"Id": book_to_index}) #.apply(lambda x: x['Id'], axis=1)

    # виділення id користувачів
    users_ids = df2['User_id'].unique()
    print(len(users_ids))
    # заміна id-string на id-int
    user_to_index = {user_id: idx for idx, user_id in enumerate(users_ids, 1)}
    df2['User_id'] = df2['User_id'].map(user_to_index) #df2.replace({"User_id": user_to_index})


    # виділення дублікатів id книжок
    book_id = df2[['Id','Title']].drop_duplicates()
    df2 = df2.sample(frac=1, random_state=42).reset_index(drop=True)
    # додаваняя поля book_id
    df1 = pd.merge(book_id, df1, on='Title', how='left')

    df1 = df1[['Id', 'Title', 'description', 'authors', 'categories']]
    # видалення пустих полів
    df1 = df1.dropna(subset=['Title'])
    df1 = df1.reset_index(drop=True)

    df2 = df2[['Id', 'User_id', 'review/score', 'review/text']]
    df2 = df2.dropna(subset=['User_id', 'review/text'])
    

    # нові назви полів
    new_col1 = {'Id'            : 'book_id',
                'Title'         : 'title', 
                'description'   : 'preview',
                'categories'    : 'genres', 
                'ratingsCount'  : 'rating'}
    
    new_col2 = {'Id'            : 'book_id',
                'User_id'       : 'user_id',
                'review/score'  : 'score',
                'review/text'   : 'text'}
    
    # зміна назви полів
    df1 = df1.rename(columns=new_col1)
    df2 = df2.rename(columns=new_col2)


    # нормалізація поля sentiment_score
    df2['sentiment_score'] = df2['score'] / 5
    # створення поля score
    df2['score'] = np.where(df2['sentiment_score'] > 0.5, 1, 0) 
    
    # створення тестових файлів
    for i in test_files:
        df2 = get_test_data(df2,i)

    # зменшення к-ті даних коментарів
    df2 = df2[:30000]

    # вивід інформації про утвлрені датафрейми
    print('BOOK_INFO:')
    print(df1.shape)
    print(df1.info())
    print(df1.isnull().sum())

    print('\nREVIEW_INFO:')
    print(df2.shape)
    print(df2.info())
    print(df2.isnull().sum())
    print('\n')

    return df1, df2


# def train_dict(text):
#     t1 = time.time()
#     d = []
#     text = list(text)
#     counter = 0
#     for sentence in text:
#         counter +=1
#         if counter % 10000 == 0:
#             print(counter)
#             print(sentence)
#         d.append(sentence.split())

#     t2 = time.time()
#     t = round((t2 - t1) / 60, 4)
#     print(f'Finished creating dictionary in {t} min.')
#     return d


def book_preprocessing(df):
    """
    Підготовка датафрейму з інформацією про книжки.
    Input: df - датафрейм з інформацією про книжки з необробленими текстовими полями.
    Output: df - оброблений датафрейм.
    """
    # обробка існуючих текстових полів
    df['preview'] = df.apply(lambda x: preprocessing(x['preview'], 0), axis=1)
    df['genres'] = df.apply(lambda x: preprocessing(x['genres'], 0), axis=1)
    df['authors'] = df.apply(lambda x: preprocessing(x['authors'], 0), axis=1)

    # створення поля description, яке поєднує в собі інформацію з
    # полів preview, genres, authors
    df['description'] = df['preview'] + ' '
    df['description'] = df['description'] + df['genres'] 
    df['description'] = df['description'] + ' '
    df['description'] = df['description'] + df['authors'] 

    return df


t1 = time.time()
# підготовка датафреймів, тестових файлів
book_info, review_info = prepare_dataframes(file1, file2, test_files)
print('files were opened')
# обробка датафреймів з інформацією про книжки
book_info = book_preprocessing(book_info)# shape (221989, 7)
book_info.to_csv(book_train_path, index=False)
print('book_info saved')
# обробка датафреймів з інформацією про відгуки
review_info['text'] = review_info.apply(lambda x: preprocessing(x['text'], 1), axis=1) # shape (999 955, 5)
review_info.to_csv(review_train_path, index=False) # 1000 - 1.21
print('review_info saved')

t2 = time.time()

# час для підготовки даних
s = 'Time for text preprocessing: {:.4f} minutes.\n'.format((t2 - t1) / 60)
# f = open('time.txt', "w+")
# f.write('Time for text preprocessing: {:.4f} minutes.\n'.format((t2 - t1) / 60))
# f.close()
write_time(s)