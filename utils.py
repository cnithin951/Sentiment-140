import re
import pandas as pd


def clean_text(text):
    """Function is used to preprocess user tweets.

    Removes @username and links, and whitespaces.

    Args:
        tweet (str): Raw text of the tweet.

    Returns:
        result (str): Processed tweet, after removing unnecessary data.
    """

    # removing @username
    result = re.sub(r'@[A-Za-z0-9]+', '', text)
    # removing link
    result = re.sub(r'https?://[A-Za-z0-9./]+', '', result)
    # removing leading and trailing whitespace
    result = result.strip()

    return result


def convert_label(polarity):
    """Simple function to preprocess polarity.
    """

    if polarity == 4:
        return 1
    elif polarity == 0:
        return 0
    else:
        print('[WARNING]')


def load_data():
    """Loads the data.

    Function loads in data, preprocesses the data
    (removes @username and links, and converts sentiment
    into neg: 0 and pos: 1)

    Returns:
        texts (pd.Series): Preprocessed texts.
        sentiment (pd.Series): Preprocessed sentiments.
    """

    train_dir = 'data\\train.csv'
    columns = ['Polarity', 'ID', 'Date', 'Query', 'User', 'Texts']

    df = pd.read_csv(train_dir, encoding='latin-1', names=columns, header=None)
    df.drop(['ID', 'Date', 'Query', 'User'], axis=1, inplace=True)

    sentiment_raw = df['Polarity']
    sentiment = sentiment_raw.apply(lambda x: convert_label(x))

    texts_raw = df['Texts']
    texts = texts_raw.apply(lambda x: clean_text(x))

    return texts, sentiment
