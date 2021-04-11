import pandas as pd
import re
import string
from nltk import word_tokenize
from nltk.corpus import stopwords

def preprocess_tweet(tweet):
    emoticons_happy = set([
        ':-)', ':)', ';)', ':o)', ':]', ':3', ':c)', ':>', '=]', '8)', '=)', ':}',
        ':^)', ':-D', ':D', '8-D', '8D', 'x-D', 'xD', 'X-D', 'XD', '=-D', '=D',
        '=-3', '=3', ':-))', ":'-)", ":')", ':*', ':^*', '>:P', ':-P', ':P', 'X-P',
        'x-p', 'xp', 'XP', ':-p', ':p', '=p', ':-b', ':b', '>:)', '>;)', '>:-)',
        '<3'
    ])
    emoticons_sad = set([
        ':L', ':-/', '>:/', ':S', '>:[', ':@', ':-(', ':[', ':-||', '=L', ':<',
        ':-[', ':-<', '=\\', '=/', '>:(', ':(', '>.<', ":'-(", ":'(", ':\\', ':-c',
        ':c', ':{', '>:\\', ';('
    ])
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)
    emoticons = emoticons_happy.union(emoticons_sad)
    # remove URLs
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))', '', tweet)
    tweet = re.sub(r'http\S+', '', tweet)
    # remove usernames
    tweet = re.sub('@[^\s]+', '', tweet)
    # remove the # in hashtag and split hashtags
    tweet_toks = tweet.split(" ")
    final_tweet_toks = []
    for i in range(len(tweet_toks)):
        if tweet_toks[i].startswith("#"):
            hashtag = tweet_toks[i]
            hashtag = hashtag[1:]
            split_hashtag = re.findall('[0-9]+|[A-Z][a-z]+|[A-Z][A-Z]+|[a-z]+', hashtag)
            final_tweet_toks = final_tweet_toks + split_hashtag
        else:
            final_tweet_toks.append(tweet_toks[i])
    tweet = " ".join(final_tweet_toks)
    # remove emojis
    tweet = emoji_pattern.sub(r'', tweet)
    # convert text to lower-case
    tweet = tweet.lower()
    #Remove any other punctuation
    tweet = [char for char in tweet if char not in string.punctuation]
    tweet = ''.join(tweet)
    #Remove non-ascii characters
    tweet = re.sub(r'[^\x00-\x7F]+', ' ', tweet)
    #Remove stopwords and emoticons from final word list
    tokens = word_tokenize(tweet)
    stop_words = set(stopwords.words('english'))
    filtered_tweet = []
    for w in tokens:
        if w not in stop_words and w not in emoticons:
            filtered_tweet.append(w)
    return ' '.join(filtered_tweet)

pre = []
file_name = ""
df = pd.read_csv(file_name, encoding='latin1', engine='python')

for row in df["tweets"]:
    preprocessed_tweet = preprocess_tweet(row)
    pre.append(preprocessed_tweet)

df["text"] = pre

df.to_csv(file_name + "_pre.csv")
