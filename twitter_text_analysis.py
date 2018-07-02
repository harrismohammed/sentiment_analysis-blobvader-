# -*- coding: utf-8 -*-
"""
Created on Sun Jun 24 14:27:47 2018

@author: harris_me
"""

#Importing the libraries

import numpy as np
import tensorflow as tf
import re
import time
import json

######## DATA PREPROCESSING ######

#Importing raw_tweet_data
#Before importing the dataset, put the json objects in an array and assign a key for easier access

x =open('pollution.json',encoding = 'utf=8',errors = 'ignore')

#Loading the raw_tweet_data
data = json.load(x)

#Getting text data from retweets and orignal tweets

tweets = []

for p in range(len(data['foo'])):
    if data['foo'][p]['lang'] == 'en':
        if len(data['foo'][p]) == 27:
            tweets.append(data['foo'][p]['text'])
        elif len(data['foo'][p]) >= 28:
            if data['foo'][p]['is_quote_status'] == 'True':
                tweets.append(data['foo'][p]['quoted_status']['extended_tweet']['full_text'])
            else:
                tweets.append(data['foo'][p]['text'])
        elif len(data['foo'][p]) >= 29:
            if data['foo'][p]['is_quote_status'] == 'True':
                tweets.append(data['foo'][p]['quoted_status']['extended_tweet']['full_text'])
            else:
                tweets.append(data['foo'][p]['text'])
      

def clean_text (text):
    text = text.lower()
    text = text.replace("rt ","")
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"\'s", " is", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"[-()\"/;:|<>{}+=#_.,']", "", text)
    return text

    
#Cleaming the tweets
clean_tweets = []

for tweet in tweets:
    clean_tweets.append(clean_text(tweet))




########### SENTIMENT ANALYSIS (V A D E R)############

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer 

analysed_tweets = []
analyser = SentimentIntensityAnalyzer()

def print_sentiment_scores(sentence):
    snt = analyser.polarity_scores(sentence)
    analysed_tweets.append("{:-<40} {}".format(sentence, str(snt)))

for clean_tweet in clean_tweets:
    print_sentiment_scores(clean_tweet)

clean_analysed_tweets = []
for analysed_tweet in analysed_tweets:
    clean_analysed_tweet = analysed_tweet.split('{')
    clean_analysed_tweet[1] = clean_analysed_tweet[1].replace("}", "").replace(" ","")
    clean_analysed_tweet[1] = clean_analysed_tweet[1].split(',')
    for l in range(len(clean_analysed_tweet[1])):
        clean_analysed_tweet[1][l] = clean_analysed_tweet[1][l].split(':')
    clean_analysed_tweet[1][0][1] = float(clean_analysed_tweet[1][0][1])
    clean_analysed_tweet[1][1][1] = float(clean_analysed_tweet[1][1][1])
    clean_analysed_tweet[1][2][1] = float(clean_analysed_tweet[1][2][1])
    clean_analysed_tweet[1][3][1] = float(clean_analysed_tweet[1][3][1])
    clean_analysed_tweets.append(clean_analysed_tweet)
 
   
positivity_in_tweet = []
negativity_in_tweet = []
neutrality_in_tweet = []

for c in range(len(clean_analysed_tweets)):
    for d in range(len(clean_analysed_tweets[c][1])):
        if (clean_analysed_tweets[c][1][d][0]) == "'neg'":
            negativity_in_tweet.append(clean_analysed_tweets[c][1][d][1])
        elif (clean_analysed_tweets[c][1][d][0]) == "'pos'":
            positivity_in_tweet.append(clean_analysed_tweets[c][1][d][1])
        elif (clean_analysed_tweets[c][1][d][0]) == "'neu'":
            neutrality_in_tweet.append(clean_analysed_tweets[c][1][d][1])
            


########### SENTIMENT ANALYSIS (T E X T B L O B)############

from textblob import TextBlob

analysed_tweets_b = []


def print_sentiment_scores_b(clean):
    analyserb = TextBlob(clean)
    sent = analyserb.sentiment
    analysed_tweets_b.append("{:-<40} {}".format(clean, str(sent)))


for clean in clean_tweets:
    print_sentiment_scores_b(clean)

clean_analysed_tweets_b = []
for analysed_tweet_b in analysed_tweets_b:
    clean_analysed_tweet_b = analysed_tweet_b.split('Sentiment')
    clean_analysed_tweet_b[1] = clean_analysed_tweet_b[1].replace("(","").replace(")","").replace(" ","")
    clean_analysed_tweet_b[1] = clean_analysed_tweet_b[1].split(',')
    clean_analysed_tweet_b[1][0] = clean_analysed_tweet_b[1][0].split('=')
    clean_analysed_tweet_b[1][1] = clean_analysed_tweet_b[1][1].split('=')
    clean_analysed_tweet_b[1][0][1] = float(clean_analysed_tweet_b[1][0][1])
    clean_analysed_tweet_b[1][1][1] = float(clean_analysed_tweet_b[1][1][1])
    clean_analysed_tweets_b.append(clean_analysed_tweet_b)
  
  
###### Converting Vader Polarity range to (-1,1)####
polarity = []

for e in range(len(positivity_in_tweet)):
    if negativity_in_tweet[e] > positivity_in_tweet[e] and negativity_in_tweet[e] > neutrality_in_tweet[e]:
        f = (2 * (positivity_in_tweet[e] + neutrality_in_tweet[e]))/3
        f = -1 + f
        f = str(f)
        polarity.append(f)
    elif positivity_in_tweet[e] > negativity_in_tweet[e] and positivity_in_tweet[e] > neutrality_in_tweet[e]:
        g = (2 * (negativity_in_tweet[e] + neutrality_in_tweet[e]))/3
        g = 1 - g
        g = str(g)
        polarity.append(g)
    elif neutrality_in_tweet[e] > negativity_in_tweet[e] and neutrality_in_tweet[e] > positivity_in_tweet[e]:
        if negativity_in_tweet[e] > positivity_in_tweet[e]:
            h = (0 - (2 * negativity_in_tweet[e])/3)
            h = str(h)
            polarity.append(h)
        else :
            i = (0 + (2 * positivity_in_tweet[e])/3)
            i = str(i)
            polarity.append(i)
           
polarity = [float(k) for k in polarity]

##### Finding the weighted average based on popularity ######

analysed_tweets_c_polarity = []

for q in range(len(polarity)):
    w1 = 0.166 #based on stars on github repo
    w2 = 0.833
    v = (w1 * polarity[q]) + (w2 * clean_analysed_tweets_b[q][1][0][1]) 
    v = str(v)
    analysed_tweets_c_polarity.append(v)

