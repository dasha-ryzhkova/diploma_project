import tkinter as tk   # from tkinter import Tk for Python 3.x
from tkinter import filedialog, messagebox
import pandas as pd
import time

import webbrowser
import os
from tensorflow.keras.preprocessing.sequence import pad_sequences
#from tensorflow.keras import Sequential
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from constants import *
from load_recomendation_data import *
from help_func import *
from recommendation import *
from processed_data import *

import warnings
warnings.filterwarnings(action='ignore')


def get_recommendation(book, user, score):
    """
    Знаходить id рекомендованих книжок.
    Input: book - id книги, до якої був залишений відгук;
           user - id користувача;
           score - оцінка, якою було оцінено відгук користувача.
    Output: list - список id рекомендованих книжок.
    """
    user_based = []
    item_based = []
    try:
        # знаходження подібних користувачів
        sim_users = find_sim_user(int(user), users_sim)
    except:
        # якщо користувач новий - утворення списку подібних користувачів
        sim_users = not_existed(book, user, score, rating, n=5)
    if len(sim_users) > 0:
        try:
            # якщо подібні користувачі знайдені, тоутвоюється стписок подібних книг
            print(f'users {sim_users}')
            user_based = users_books(sim_users, rating, n=5)
        except:
            print('Not enough data!')
    try:
        # створюється список подібних елементів на основі вмісту
        item_based = find_similar_books(int(book), content_based_data, emb_vec)
    except Exception as e:
        print('Something wrong!')
        print(e)
    print('user_based: ', user_based)
    print('item_vased: ', item_based)
    # повертається список унікальних id
    return list(set(user_based + item_based))


def create_recomendation_list(input_file, output_file):
    """
    Створює рекомендації для користувачів на основі відгуків.
    Input: input_file - шлях до вхідного файлу;
           output_file - шлях до результуючого файлу.
    Output: result - результуючий датафрк=ейм з полями 'user_id', 'book_id', 'title'.
    """
    df = pd.read_csv(input_file)
    print(df)
    # попередня обробка тексту відгуку
    X_test = df.apply(lambda x: preprocessing(x['text'], 1), axis=1)
    X_test = pad_sequences(tokenizer.texts_to_sequences(X_test), maxlen=DATA_LEN)
    # надання оцінки тональності відгуку
    y_pred = sent_model.predict(X_test)
    df['sentiment_score'] = y_pred
    df['score'] = np.where(df['sentiment_score']>0.5, 1, 0)
    # відбираються позитивні відгуки
    pos_rev = df[ df['score'] == 1]
    #print(pos_rev)

    recommend_books_idx = []
    titles = []

    ids_books = list(pos_rev['book_id'])
    ids_users = list(pos_rev['user_id'])
    # створення списку кортежів (book_id, user_id)
    ids = set(zip(ids_books, ids_users))
    #print(ids)
    t1 = time.time()
    # створення списку рекомендацій для кожного користувача
    for book, user in ids:
        print(book, user)
        t = time.time()
        score = float(pos_rev.loc[pos_rev['book_id'] == int(book)]['sentiment_score'])
        # список id рекомендованих книг
        books_list = get_recommendation(book, user, score)
        titles_list = []
        # знаходження назви книги за id
        for book in books_list:
            title = book_train.loc[book_train['book_id'] == int(book)]['title'].item()
            print(title)
            titles_list.append(title)
        #     
        # titles_list = list(set(titles_list))
        recommend_books_idx.append(books_list)
        titles.append(titles_list)
        print(f'for user_id={user} recommendation created for {time.time()-t} sec')

    # створення результуючого датафрейму
    print(f'recommendations for {time.time()-t1} sec')
    result = pd.DataFrame({'user_id': ids_users,
                           'book_id': recommend_books_idx,
                           'title'  : titles})
    result.to_csv(output_file)
    return result 

def show_dataframe(df, html_file='dataframe.html'):
    """
    Виводить результуючий файл в окреме вікно веб браузеру.

    Input: df - датафрейм з результатами роботи програми;
           html_file - файл для запису html представлення таблиці
    """
    # обирання полів для представлення
    df = df[['user_id', 'book_id', 'title']]
    df = df.set_index('user_id')
    # конвертація датафрейму в html-сторінку
    html_str = df.to_html()
    # збереження html файлу
    with open(html_file, 'w') as f:
        f.write(html_str)

    abs_path = os.path.abspath(html_file)
    # відкриття html-сторінки у веб-браузері
    webbrowser.open(f'file://{abs_path}', new=2)

# створення класу додатку
class App:
    def __init__(self, root):
        self.root = root
        self.root_title = 'some title'

        #self.root.configure(bg='#35155D')
        # ініціація основного вікна
        # ініціація параметрів
        self.root.geometry("800x600")
        self.font = ('Helvetica', 16)

        self.label_1 = tk.Label(root, text='Recommendation System', font=self.font)
        self.label_1.pack(pady=5)
        # ініціація кнопки вибору вхідного файлу
        self.choice_button = tk.Button(root, text='Open file', command=self.open_file, font=self.font)
        self.choice_button.pack(pady=5)

        self.label_2 = tk.Label(root, text='Enter result file', font=self.font)
        self.label_2.pack(pady=5)
        # ініціація кпоял для введення шляху до результуючого файлу
        self.entry = tk.Entry(root)
        self.entry.pack(pady=5)
        # ініціація кнопки початку роботи програми
        self.start_button = tk.Button(root, text='Create recommendations', command=self.start, font=self.font)
        self.start_button.pack(pady=5)
        # ініціація кнопки вибору виводу результатів
        self.show_button = tk.Button(root, text='Open result', command=self.open_res, state=tk.DISABLED, font=self.font)
        self.show_button.pack(pady=5)

        self.file = ''
        self.result = None


    def open_file(self):
        # відкриття вхфдного файлу
        self.file = filedialog.askopenfilename(title = "Select file",filetypes = (("CSV Files","*.csv"),))
        if self.file:
            messagebox.showinfo("Selected File", f"You selected: {self.file}")
            print(self.file)



    def start(self):
        try:
            # вилучення шляху до результуючого файлу
            self.output_file = self.entry.get()
            if not self.file:
                messagebox.showwarning("No File", "Please select an input file first.")
                return

            if not self.output_file:
                messagebox.showwarning("No Output File", "Please enter an output file name.")
                return
            # перевірка назви вихідного файлу
            if not self.output_file.endswith('.csv'): # or re.match(r'[^a-zA-Z]', self.output_file) :
                messagebox.showwarning("Not Correct Output File", "Please enter an output .csv file name.")
                return
            # створення списку рекомендацій
            self.result = create_recomendation_list(self.file, self.output_file)
            messagebox.showinfo("Program Finished", "The program has finished processing.")
            self.show_button.config(state=tk.NORMAL, font=self.font)
        except Exception as e:
            print(e)
            messagebox.showerror("Error", str(e))

    def open_res(self):
        if self.result is not None:
            # виведення утвореного датафрейму у вікно веб-браузера
            show_dataframe(self.result)
        else:
            messagebox.showwarning("No Data", "No data to display. Please run the program first.")














