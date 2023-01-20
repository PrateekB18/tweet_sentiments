# -*- coding: utf-8 -*-
"""
##########################
Created on Wed Jan 11 2023
@author: Prateek
##########################
"""

import tweepy
import credentials
import pandas as pd
import re
import datetime
import time
import textblob as tb
import numpy as np
import nltk 

auth = tweepy.OAuthHandler(credentials.CONSUMER_KEY, credentials.CONSUMER_SECRET)
auth.set_access_token(credentials.ACCESS_TOKEN, credentials.ACCESS_TOKEN_SECRET)

api = tweepy.API(auth,wait_on_rate_limit=True)

# Function to fetch tweets 
def fetch_tweets(keyword1, keyword2 = None, count = 10000, since = None):

    today = datetime.datetime.now()
    today = today.replace(hour=23, minute=59, second=59, microsecond=999999) # set from the beggining of the day
    time_to_the_past = 1 # 1 because we want 1 day before today
    yesterday = today - datetime.timedelta(time_to_the_past) 
    
    # Collecting tweets
    count = count # Set the number of tweets to retrieve
    next_day = yesterday + datetime.timedelta(time_to_the_past)
    
    if keyword2 == None:
        search = f'{keyword1} -filter:retweets'
    else:
        search = f'{keyword1} OR {keyword2} -filter:retweets'
    
    tweets = tweepy.Cursor(api.search_tweets, q=search, lang = 'en', 
                           since = since,
                           until = next_day.date(),
                           tweet_mode = 'extended').items(count)
    
    tweets_list = []
    for tweet in tweets:
                # Filtering by date
                if  yesterday.date() == tweet.created_at.date(): 
                    full_text = tweet.full_text
                    tweets_list.append([tweet.user.id,
                                        tweet.id_str,
                                        full_text,
                                        tweet.user.verified,
                                        tweet.favorite_count,
                                        tweet.created_at,
                                        tweet.user.location,
                                        tweet.user.friends_count,
                                        tweet.user.followers_count
                                        ])
    df = pd.DataFrame(tweets_list, columns=['UserID', 'ID', 'Tweets', 'Verified', 'Likes', 
                                            'Time', 'Location',
                                            'Following', 'Followers'])
    df = df.drop_duplicates(subset=['ID'], keep='first')
    since = df.iloc[-1]['Time']
    return since,df

# Function for cleaning text in tweets
def cleanTwt(twt):
    if type(twt) == np.float:
        return ""
    twt = twt.lower()
    twt = re.sub("'", "", twt) # to avoid removing contractions in english
    twt = re.sub("@[A-Za-z0-9_]+","", twt)
    twt = re.sub("#[A-Za-z0-9_]+","", twt)
    twt = re.sub(r'http\S+', '', twt)
    twt = re.sub('[()!?]', ' ', twt)
    twt = re.sub('\[.*?\]',' ', twt)
    twt = re.sub("[^a-z0-9]"," ", twt)
    return twt

# Function for determining subjectivity using TextBlob
def detSubjectivity(text):
    return tb.TextBlob(text).sentiment.subjectivity

# Function for determining polarity using TextBlob
def getPolarity(text):
    return tb.TextBlob(text).sentiment.polarity

# Function for further cleaning and tokenizing text in tweets
def tokenization(text):
    ps = nltk.PorterStemmer()
    stopword = nltk.corpus.stopwords.words('english')
    text = re.split('\W+', text)
    text = [word for word in text if word not in stopword]
    text = [ps.stem(word) for word in text]
    text = list(filter(None, text))
    return text

# Function for joining tokenized tweets
def joined(text):
    text = " ".join(word for word in text)
    return text

# Iterative tweet fetching
since_old, data = fetch_tweets('Crypto')
starttime = time.time()
mint = 15
i=0
while True:
    i = i+1;
    print(f'Iteration {i}')
    since, data_new = fetch_tweets('Crypto', since_old)
    since_old = since
    data = data.append(data_new)
    time.sleep(mint*60.0 - ((time.time() - starttime) % mint*60.0)) #Sleep for 15 minutes (API limit)
    if i==10:
        break

data.drop(['Likes','Time','Location','Following','Followers', 'Verified'], axis=1, inplace=True)
data['CleanTwt'] = data['Tweets'].apply(cleanTwt)
data['TokenizedTwt'] = data['CleanTwt'].apply(lambda x: tokenization(x.lower()))
data['Tokens'] = data['TokenizedTwt'].str.len()
data = data[data.Tokens >15]
data.drop(['Tokens'], axis=1,inplace=True)
data['JoinedTwt'] = data['TokenizedTwt'].apply(lambda x: joined(x))

data['subjectivity'] = data['Tweets'].apply(detSubjectivity)
data['polarity'] = data['Tweets'].apply(getPolarity)

data.reset_index(drop=True, inplace=True)
data.to_csv('Data.csv')

