from utils import clean_text
import pickle

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

MAX_SEQ_LEN = 117

print('[INFO] Loading tokenizer...')
with open('model\\tokenizer.pickle', 'rb') as f:
    tokenizer = pickle.load(f)

print('[INFO] Loading model...')
model = load_model('model\\model.h5')

while True:
    text = input('\nEnter your text "q" to exit: ')
    if text == 'q':
        break

    text = clean_text(text)
    sequence = tokenizer.texts_to_sequences(text)
    data = pad_sequences(sequence, maxlen=MAX_SEQ_LEN)

    y_pred = model.predict_classes(data)[0]
    sentiment = 'Positive' if y_pred == 1 else 'Negative'

    print(f'\n\t Text: {text}')
    print(f'\t Sentiment: {sentiment}')
