import re


def clean_text(text):
    """Function is used to preprocess user tweets.

    Removes @username and links.

    Args:
        tweet (str): Raw text of the tweet.

    Returns:
        result (str): Processed tweet, after removing unnecessary data.
    """

    # removing @username
    result = re.sub(r'@[A-Za-z0-9]+', '', text)
    # removing link
    result = re.sub(r'https?://[A-Za-z0-9./]+', '', result)

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
