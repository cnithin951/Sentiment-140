import numpy as np
import time
import pickle
import os
import seaborn as sns
import matplotlib.pyplot as plt

from utils import load_data

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D
from tensorflow.keras.layers import Dense, Flatten, Embedding
from tensorflow.keras.layers import GlobalMaxPooling1D, Dropout

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

VOCAB_SIZE = 20000
MAX_SEQ_LEN = 256
EMB_DIM = 50
BASE_DIR = str(int(np.ceil(time.time())))

if not os.path.exists(BASE_DIR):
    os.makedirs(BASE_DIR)
    print(f'[INFO] {BASE_DIR} created')

print('[INFO] Loading data...')
texts, sentiments = load_data()

train_texts, val_texts, train_sent, val_sent = train_test_split(texts,
                                                                sentiments,
                                                                test_size=0.2)

tk = Tokenizer(num_words=VOCAB_SIZE)
tk.fit_on_texts(train_texts)

with open(f'{BASE_DIR}\\tokenizer.pickle', 'wb') as f:
    pickle.dump(tk, f)

word_index = tk.word_index
print('[INFO] Number of unique tokens found (in train data):', len(word_index))

x_train = tk.texts_to_sequences(train_texts)
x_test = tk.texts_to_sequences(val_texts)

max_length = len(max(x_train, key=len))
if max_length > MAX_SEQ_LEN:
    max_length = MAX_SEQ_LEN

x_train = pad_sequences(x_train, maxlen=max_length)
x_test = pad_sequences(x_test, maxlen=max_length)
y_train = np.array(train_sent).reshape(-1, 1)
y_test = np.array(val_sent).reshape(-1, 1)

print(f'[INFO] Sequence Length: {max_length}')
print(f'[INFO] Shape of x_train: {x_train.shape}')
print(f'[INFO] Shape of y_train: {y_train.shape}')
print(f'[INFO] Shape of x_test: {x_test.shape}')
print(f'[INFO] Shape of y_test: {y_test.shape}')


print('[INFO] Indexing word vectors...')
embeddings_index = {}
embedding_path = f'C:\\Users\\gauta\\Python Projects\\Pretrained\\glove.6B.{EMB_DIM}d.txt'

with open(embedding_path, encoding='utf8') as f:
    for line in f:
        word, coefs = line.split(maxsplit=1)
        coefs = np.fromstring(coefs, 'f', sep=' ')
        embeddings_index[word] = coefs

print('[INFO] Total number of word vectors in Glove Embedding:', len(embeddings_index))

print('[INFO] Preparing embedding matrix...')
num_words = min(VOCAB_SIZE, len(word_index) + 1)
embeddings_matrix = np.zeros((num_words, EMB_DIM))  # initializing zeros matrix

for word, i in word_index.items():
    if i >= VOCAB_SIZE:
        continue

    embedding_vector = embeddings_index.get(word)  # vector for that word
    if embedding_vector is not None:  # if word not found, then 0
        embeddings_matrix[i] = embedding_vector

embedding_layer = Embedding(num_words, EMB_DIM,
                            weights=[embeddings_matrix],
                            input_length=max_length,
                            trainable=True)

model = Sequential()

model.add(Input(shape=(max_length,)))
model.add(embedding_layer)

model.add(Dropout(0.2))

model.add(Conv1D(250, 3, activation='relu', padding='valid'))
model.add(GlobalMaxPooling1D())

model.add(Dense(250, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

print(model.summary())

plot_model(model, to_file=f'{BASE_DIR}\\model.png', show_shapes=True,
           dpi=200, expand_nested=True)

es = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)

model.fit(x_train, y_train, batch_size=32, validation_split=0.2,
          epochs=1000, callbacks=[es])

model.save(f'{BASE_DIR}\\model.h5')

loss, acc = model.evaluate(x_test, y_test, verbose=0)
print(f'\n\t[INFO] Accuracy: {acc} Loss: {loss}')

y_pred = model.predict_classes(x_test)

matrix = confusion_matrix(y_test, y_pred)
matrix = matrix / matrix.astype(np.float).sum(axis=0)
sns.heatmap(matrix, annot=True, cmap='coolwarm')
plt.savefig(f'{BASE_DIR}\\cm.png')

os.rename(BASE_DIR, BASE_DIR + '_loss_' + str(loss) + '__accuracy__' + str(acc))
print('\n\t[DONE]')
