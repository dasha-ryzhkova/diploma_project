import emot
import re

from constants import *

def replace_emoji(text):
    """
    Заміна емоджі-символів на їхній словесний опис.
    Input: text - вхідний текст з емоджі-символів.
    Output: text - оброблений текст.
    """

    emot_object = emot.core.emot()
    meanings = emot_object.emoji(text)
    # перевірка тексту на наявність емоджі-символів
    if meanings['flag'] is False:
        return text
    # заміна емоджі-символів на іххій словесний опис
    while meanings['flag']:
        emoji = meanings['mean'][0][1:-1].replace('_', ' ')
        idx = meanings['location'][0][0]
        start = text[:idx]
        end = text[idx + 1:]
        text = ''.join([start, emoji, end])
        meanings = emot_object.emoji(text)

    return text


def preprocessing(text, emoji=1):
    """
    Обробка тестових даних. Преведення до нижнього регістру, видалення
    небуквенних сиволів, коротких та стоп-слів, html-тегів.
    Input: text - вхідний необроблений текст.
           emoji - прапорець, що відповідає за виклик 
           функції replace_emoji(text), яка викликається тільки для 
           обробки коментарів.
    Output: text - оброблений текст.
    """
    # перевірк на пустий рядок    
    if type(text) is float:
        #return None
        text = 'book' 

    # перевірка чи текст є коментарем для виклику replace_emoji(text)
    if emoji:
        text = replace_emoji(text)

    # переведення тексту в нижній регістр       
    text = text.lower()

    # видалення html-тегів
    tag_pattern = r'<[^>]*>'
    text = re.sub(tag_pattern, ' ', text)

    # видалення небуквенних символів
    symb_pattern = r'[^a-zA-Z]'
    text = re.sub(symb_pattern, ' ', text)

    # розділення тексту на слова
    words = text.split(' ')
    new_words = []
    for word in words:   
        # пропуск стоп-слів
        if word in STOP_WORDS:
            continue
        # пропуск слів, коротших за 3 символи
        if len(word) < 3:
            continue

        new_words.append(word)
    # вибрані слова перетворюються на рядок
    new_data = ' '.join(new_words)
    return new_data #pd.Series(preprocessed)


def write_time(s, file='time.txt'):
    """
    Запис часу виконання завдання у файл.
    Input: s - рядок, який буде записаний до файлу.
           file - файл для запису.
    """
    with open(file, 'a') as f:
        f.write(s)
