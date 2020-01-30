import numpy as np
import pandas as pd
import nltk
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils import convert_label, clean_text

train_dir = 'data\\train.csv'
columns = ['Polarity', 'ID', 'Date', 'Query', 'User', 'Texts']

df = pd.read_csv(train_dir, encoding='latin-1', names=columns, header=None)
df.drop(['ID', 'Date', 'Query', 'User'], axis=1, inplace=True)

sentiment_raw = df['Polarity']
sentiment = sentiment_raw.apply(lambda x: convert_label(x))

texts_raw = df['Texts']
texts = texts_raw.apply(lambda x: clean_text(x))

len_tweets = [len(s.split()) for s in texts]
len_tweets.sort()
words_per_sample = np.median(len_tweets)
s_w = len(sentiment) / words_per_sample

print(f'\n[INFO] Number of Samples: {len(sentiment)}')
print(f'[INFO] Number of classes: {len(sentiment.unique())}')
print(f'[INFO] Number of samples per class:')
print(f'{sentiment.value_counts()}')
print(f'\n[INFO] Number of words per sample (median): {words_per_sample}')
print(f'\n[INFO] Number of samples by number of words per sample: {s_w}')

if s_w < 1500:
    print('[INFO] Convert data into n-grams and use MLP.')
else:
    print('[INFO] Convert data into sequences and use sepCNN.\n')

print('[INFO] Tokenizing...')
total_words = []
for tweet in tqdm(texts):
    total_words.extend(tweet.split())

print('[INFO] Calculating frequency distribution...')
fd = nltk.FreqDist(total_words)

plt.title('Frequency Distribution of top 100 n-grams')
fd.plot(100, cumulative=False)
plt.show()

plt.title('Sample Length Distribution')
plt.hist(len_tweets, 50)
plt.xlabel('Length of a sample')
plt.ylabel('Number of samples')
plt.savefig('Sample_len_dist.png')
plt.show()
