import numpy as np
import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity

from constants import *
#from load_recomendation_data import content_based_data
from help_func import *

# ITEM BASED RECOMMENDATOR
def similiraty_of_text(text1, text2, model):
    """
    Знаходження векторноі подібності між двома текстами
    Input: text1 - перший текст; 
           text2 - другий текст; 
           model - навчена модель Word2Vec.
    Output: similiraty - подібность між двома текстами
    """
    # розбиття тексту на слова 
    text1 = text1.split(' ') # preprocessing(text1, 0).split(' ')
    text2 = text2.split(' ') # preprocessing(text2, 0).split(' ')

    # якщо тексти однакові - коефіцієнт подібності = 1
    if text1 == text2:
        return 1
    
    vector1 = np.zeros(VECTOR_SIZE)
    vector2 = np.zeros(VECTOR_SIZE)
    # заповнення першого вектора
    for word in text1:
        try:
            vector1 = np.add(vector1, model[word])
        except Exception as e:
            # якщо слова немає в словнику - вектор випадкових чисел
            vector1 = np.add(vector1, np.random.rand(VECTOR_SIZE))
    # заповнення другого вектора
    for word in text2:
        try:
            vector2 = np.add(vector2, model[word])
        except Exception as e:
            # якщо слова немає в словнику - вектор випадкових чисел
            vector2 = np.add(vector2, np.random.rand(VECTOR_SIZE))

    # розрахунок подібності текстів
    similiraty =  np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
    return similiraty



def find_similar_books(idx, df, model, n=5):
    """
    Знаходження подібних книг
    Input: idx - id книги, до яккої шукаються подібні; 
           df - датафрейм, що містить поля "book_id" та "description"; 
           model - навчена модель Word2Vec;
           n - кількість рекомендацій.
    Output: recomendation - список id рекомендованих книг.
    """
    similarities = {}
    book_idx = list((df.index.values))
    # знаходження подібності між заданою книгою та усіма іншими
    for i in book_idx:
        similarities[i] = similiraty_of_text(df['description'].loc[idx], df['description'].loc[i], model)

    recomendation = []
    c = 0
    # сортування за спаданням подібності
    similarities = dict(sorted(similarities.items(), key=lambda item: item[1], reverse=True))
    
    for k,v in similarities.items():
        # якщо подібність = 1 - та сама книга
        if v == 1:
            continue
        # якщо подібність = о - подальший пошук не має сенсу
        if v == 0:
            break
        # знайдено 5 рекомендацій
        if c == n:
            break
        # запобігання додавання занадто схожих елементів
        if v < 0.95:
            recomendation.append(int(k))
            c +=1
        
    return recomendation


# USER BASED RECOMENDATOR

def find_sim_user(user_id, users_sim, n=5):
    """
    Знаходження подібних користувачів
    Input: user_id - id книги, до яккої шукаються подібні; 
           users_sim - матриц подібностей користувачів між собою; 
           n - кількість подібних користувачів.
    Output: users - список id подібних користувачів.
    """
    # виділення з матриці подібність заданого користувача до інших
    sim_matrix = users_sim[user_id:user_id + 1].drop(str(user_id), axis=1).to_dict(orient='records')[0]
    sim_users = {}
    c = 0
    # сортування за спаданням подібності
    sim_matrix = dict(sorted(sim_matrix.items(), key=lambda item: item[1], reverse=True))
    for k,v in sim_matrix.items():
        # якщо подібність = 1 - той самий користувач
        if v == 1:
            continue
        # знайдено 5 подібних користувачів
        if c == n:
            break
        sim_users[k] = v
        c +=1
    users = list(sim_users.keys())
    return users


def users_books(users, rating, n=5):
    """
    Знаходження подібних користувачів
    Input: users - список id подібних користувачів; 
           rating - рейтингова матриця; 
           n - кількість рекомендацій.
    Output: recomend_books - список id рекомендованих книг.
    """
    recomend_books = []
    for user in users:
        idx = int(user)
        r = rating.loc[idx].to_dict() 
        c = 0
        for k,v in r.items():
            # знайдено 5 рекомендацій
            if c == n: 
                break
            # якщо подібність = о - подальший пошук не має сенсу
            if v == 0:
                break
            # якщо подібність > о створити рекомендацію
            if v > 0:
                recomend_books.append(int(k))

    return recomend_books


def not_existed(book, user, score, rating, n=5):
    """
    Знаходження подібних користувачів для користувача, для якого немає
    подібності в підготовлених даних.  
    Input: book - id книги; 
           user - id користувача; 
           score - оцінка користувача для книги.
           n - кількість подібних користувачів.
    Output: users - список id подібних користувачів.
    """
    # створення нового запису про користувача
    book = str(book)
    new_user_rating = {book: score}
    new_user_id = user
    new_user = pd.Series(new_user_rating, name=new_user_id)
    # додаваня нового користувача в рейтингову матрицю
    rating = pd.concat([rating, pd.DataFrame([new_user])]) 
    # заповнення порожніх значень
    rating = rating.fillna(0)
    # оцінка користувача до заданої книги
    try:
        new_user_vector = rating.loc[new_user_id, book].values.reshape(-1, 1)
    except: #
        new_user_vector = rating.loc[new_user_id, book].reshape(-1, 1)
    # оцінка існуючих користувачів для заданої книги
    existing_users_vectors = rating.loc[:, book].values.reshape(-1, 1)
    # знаходження косинусної подібності для нового користувача з 
    # іншими користувачами
    similarity_scores = cosine_similarity(existing_users_vectors, new_user_vector).flatten()
    similarity_df = pd.DataFrame(similarity_scores, index=rating.index, columns=[new_user_id])
    # виділення n найбільш подібних користувачів
    similar_users = similarity_df.drop(index=new_user_id).nlargest(n, new_user_id)
    users = []
    similar_users = similar_users.to_dict()[user]
    for k, v in similar_users.items():
        # якщо подібність = о - подальший пошук не має сенсу
        if v == 0:
            break
        users.append(int(k))
    #idx = list(similar_users.index)
    return users


