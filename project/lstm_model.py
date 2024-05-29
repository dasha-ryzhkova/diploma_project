import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import os
matplotlib.style.use('bmh')

from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns


from gensim.models import Word2Vec
import time

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' 
import tensorflow as tf

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import Sequential
from tensorflow.keras.metrics import Precision, AUC
from tensorflow.keras.layers import Bidirectional, GlobalMaxPool1D, Dense, LSTM, Conv1D, Embedding
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.model_selection import train_test_split


import pickle

from constants import *
from help_func import *
from processed_data import review_train

class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    '''
    Ініціація класу Callback для обмеження епох навчання.

    Input: epoch - епоха
           logs  - значення метрики в кінці навчальної епохи 
    '''
    # Перевірка умови для припинення навчання
    if(logs.get('loss') < 0.1):
      # Зупинка навчання
      print("\nLoss is lower than 0.1 so cancelling training!")
      self.model.stop_training = True


def plotting(history):   
    '''
    Побудова графіків за ресультатами навчання моделі LSTM.

    Input: history - історія навчання моделі 
    '''

    # правильность моделі
    acc = history.history['accuracy']
    val_acc  = history.history['val_accuracy']

    # епохи навчання
    epochs   = range(len(acc)) # Get number of epochs

    # побудова графіку правильності
    plt.figure(figsize=(5,5))
    plt.plot  ( epochs,     acc, label='Правильні відповіді на навчальному датасеті' )
    plt.plot  ( epochs, val_acc, label='Правильні відповіді на тренувальному датасеті' )
    plt.xlabel('Епоха')
    plt.ylabel('Частка правильних відповідей')
    plt.title ('Точність')
    plt.ylim([0, 1])
    plt.legend(loc='best')
    plt.savefig('accuracy.png')


def conf_matrix(y_pred, y_true):
    '''
    Побудова матриці помилок моделі LSTM.

    Input: y_pred - передбачені моделлю значення; 
           y_true - справжні значення.
    '''
    # матриця помилок 
    matrix = confusion_matrix(y_true, y_pred)
    print(classification_report(y_true, y_pred))
    print(matrix)

    categories  = ['Negative','Positive']
    names = ['True Neg','False Pos', 'False Neg','True Pos']

    # побудова графіку матриці помилок
    percentages = ['{0:.2%}'.format(value) for value in matrix.flatten() / np.sum(matrix)]

    labels = [f'{v1}\n{v2}' for v1, v2 in zip(names,percentages)]
    labels = np.asarray(labels).reshape(2,2)

    plt.figure(figsize=(5,5))
    sns.heatmap(matrix, annot = labels, cmap = 'rocket',fmt = '',
                xticklabels = categories, yticklabels = categories)

    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title ("Confusion Matrix")
    plt.savefig('conf_matrix.png')

# Ініціація нового Callback
callbacks = myCallback()

# очищення датафрейму від рядків з пустими текстовими полями
review_train = review_train.dropna(subset=['text'])



t1 = time.time()

# розділення даних 
X_data = review_train['text']
y_data = review_train['score']

# ініціація та навчання Tokenizer 
tokenizer = Tokenizer(filters="", lower=False, oov_token="<oov>")
tokenizer.fit_on_texts(X_data)
tokenizer.num_words = VOCAB_LEN
print("Tokenizer vocab length:", VOCAB_LEN)

# розділення даних на тренувальні та тестові вибірки
X_train, X_test, y_train, y_test = train_test_split(X_data, 
                                                    y_data,
                                                    test_size = 0.25, 
                                                    random_state = 0)

# привелення речень до єдиної довжини  
X_train = pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=DATA_LEN)
X_test = pad_sequences(tokenizer.texts_to_sequences(X_test), maxlen=DATA_LEN)
t2 = time.time()

# збереження Tokenizer
with open(tokenizer_path, 'wb') as f:
    pickle.dump(tokenizer, f, protocol=pickle.HIGHEST_PROTOCOL)

s = 'Time for Tokenizer training: {:.4f} minutes.\n'.format((t2 - t1) / 60)
write_time(s)
# f = open('time.txt', "w")
# f.write('Time for Tokenizer training: {:.4f} minutes.\n'.format((t2 - t1) / 60))
# f.close()

# побудова матриці вбудування слів
t1 = time.time()
# завантаження моделі Word2Vec
w2v_model = Word2Vec.load(w2v_path)
embedding_matrix = np.zeros((VOCAB_LEN, VECTOR_SIZE))
vocab = w2v_model.wv.index_to_key

for word, token in tokenizer.word_index.items():
    if word in vocab:
        # запис векторного представлення слова у матрицю
        embedding_matrix[token] = w2v_model.wv.__getitem__(word)
print("Embedding Matrix Shape:", embedding_matrix.shape)

t2 = time.time()
s = 'Time for creating Embedding Matrix: {:.4f} minutes.\n'.format((t2 - t1) / 60)
write_time(s)
# f = open('time.txt', "w")
# f.write('Time for creating Embedding Matrix: {:.4f} minutes.\n'.format((t2 - t1) / 60))
# f.close()


# ініціація шару вбудування слів для моделі LSTM
embedding_layer = Embedding(input_dim = VOCAB_LEN,
                            output_dim = VECTOR_SIZE,
                            weights=[embedding_matrix],
                            input_length=DATA_LEN,
                            trainable=False)

# ініціація моделі LSTM
sent_model = Sequential([
                        embedding_layer,
                        LSTM(64, dropout=0.3, return_sequences=True),
                        LSTM(64, dropout=0.3,  return_sequences=True),
                        Conv1D(100, 5, activation='relu'),
                        GlobalMaxPool1D(),
                        Dense(16, activation='relu'),
                        Dense(1, activation='sigmoid'),
    ],
    name="LSTM_Model")
sent_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])



t1 = time.time()
# тренування моделі LSTM
history = sent_model.fit(X_train, y_train,
                        batch_size=1024,
                        epochs=EPOCHS_SENT,
                        validation_split=0.25,
                        verbose=1,
                        callbacks=[callbacks])

t2 = time.time()
# збереження моделі LSTM
sent_model.save(sent_path)
s = 'Time for LSTM training: {:.4f} minutes.\n'.format((t2 - t1) / 60)
write_time(s)
# f = open('time.txt', "w")
# f.write('Time for LSTM training: {:.4f} minutes.\n'.format((t2 - t1) / 60))
# f.close()

# побудова графіку правильності моделі LSTM
plotting(history)
# побудова графіку матриці помилок моделі LSTM
y_pred = sent_model.predict(X_test)
y_pred = np.where(y_pred>=0.5, 1, 0)
conf_matrix(y_pred, y_test)  