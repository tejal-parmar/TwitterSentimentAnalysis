#!/usr/bin/env python
# coding: utf-8

# # Twitter Sentiment Analysis

# ## Capstone Project by Group-6

# ### Importing Required Libraries

# #### <font color='blue'>Tweets Scrapping </font>

# In[1]:


import snscrape.modules.twitter as sntwitter


# #### <font color='blue'>Data Structures</font>

# In[2]:


import pandas as pd
pd.options.display.max_colwidth = 1000
import numpy as np


# #### <font color='blue'>Text Processing</font>

# In[3]:


import re
import spacy
nlp = spacy.load("en_core_web_sm")

import nltk
from nltk.corpus import stopwords

import cleantext

from better_profanity import profanity 


# #### <font color='blue'>ML Modeling</font>

# In[4]:


from transformers import pipeline

sentiment_classifier = pipeline("sentiment-analysis",
                model = "cardiffnlp/twitter-roberta-base-sentiment-latest")


# #### <font color='blue'>Plotting</font>

# In[5]:


from wordcloud import WordCloud
import matplotlib.pyplot as plt
import plotly.graph_objects as go


# #### <font color='blue'>User Interface</font>

# In[26]:


import streamlit as st

import cleantext
import time
from datetime import datetime


# ### Building Dataset

# In[7]:


def scrape_tweets(search_item):

    tweets = []

    for i, tweet in enumerate(sntwitter.TwitterSearchScraper(search_item).get_items()):
        if i > 200:
            break
        tweets.append([tweet.date, tweet.id, tweet.content, tweet.user.username])

    tweets_df = pd.DataFrame(tweets, columns=['Datetime', 'Tweet Id', 'Text', 'Username'])
    df = tweets_df[['Text']]
    df.columns = ['tweets']
    return df


# ### Data Cleaning

# #### Parsing using Spacy <br>Flitering punctuation, white space, numbers, URL, @ mention using Spacy <br>Removing special character and search item <br>Single syllable token removal <br>Spell correction: dealing with repeated characters such as "soooo gooood", if the same character is repeated more than two times, it shortens the repeatition to two. Hence "soooo gooood" will be transformed to "soo good". It will help to reduce feature space.

# In[8]:


def spacy_cleaner(text, search_item):
    parsed = nlp(text.lower())
    final_tokens = []
    for t in parsed:
        if t.is_punct or t.is_space or t.like_num or t.like_url or str(t).startswith('@'):
            pass
        else:
            sc_removed = re.sub("[^a-zA-Z]", '', str(t.lemma_))
            si_removed = sc_removed.replace(search_item, '')
            if len(si_removed) > 1:
                final_tokens.append(si_removed)

    joined = ' '.join(final_tokens)
    spell_corrected = re.sub(r'(.)\1+', r'\1\1', joined)
    return spell_corrected


# #### Removing empty rows

# In[9]:


def data_cleaning(data, search_item):
    data['clean_text'] = [spacy_cleaner(t, search_item) for t in data.tweets]
    data['clean_text'] = data['clean_text'].replace('', np.nan)
    cleaned_data = data.dropna(subset = ['clean_text'])
    cleaned_data.reset_index(drop = True,inplace = True)
    return cleaned_data


# ### Word Cloud

# #### A word cloud represents word usage in a document by resizing individual words proportionally to its frequency, and then presenting them in a random arrangement.

# In[10]:


def create_cloud(cleaned_data, search_item):
    stp_words = stopwords.words('english')
    stp_words.append('amp')
    joined = ' '.join(word for word in cleaned_data['clean_text'])
    consolidated = ' '.join(word for word in joined.split() if not profanity.contains_profanity(word))
    wordCloud = WordCloud(width = 1600, height = 800,
                          max_font_size = 200, max_words = 100,
                          background_color = 'White', collocations = False, stopwords = stp_words).generate(consolidated)
    plt.imshow(wordCloud, interpolation = 'bilinear')
    plt.axis('off')
    file_name = search_item + "_WordCloud.png"
    wordCloud.to_file(file_name)
    return plt


# ### Modeling

# #### After testing 4 popular sentiment analysis libraries in Python: NLTK, TextBlob, Flair, and HuggingFace transformers. <br> We manually validated the predicted labels for each tweet and HuggingFace transformers (Twitter_roberta) proven to be the most accurate model for our analysis.

# #### The following code uses twitter_roberta model to classify the sentiment of each tweet in the dataset, the resulted dataframe will have an additional column containing the predicted label for that particular tweet.

# In[11]:


def modeling(cleaned_data, search_item):
    results = cleaned_data.copy()
    label, score = list(), list()
    for tweet in results['clean_text']:
        sentiment = sentiment_classifier(tweet)
        label.append(sentiment[0]['label'])
        score.append(sentiment[0]['score'])
    
    results['Label'] = label
    results['Score'] = score
    
    file_name = search_item + "_clean_text_tweet_results.csv"
    results.to_csv(file_name)
    return results


# ### Plotting the Results

# In[12]:


def plt_segmentation(col, results):
    seg_df = results.pivot_table(index = [col], aggfunc = {col:'count'})
    mylabels = np.sort(results[col].unique())
    if len(mylabels) > 2:
        colors = ['#E44144', '#7CADD2', '#509B51']
    else:
        colors = ['#E44144', '#509B51']
    plt.pie(np.array(seg_df).ravel(), labels = mylabels, autopct = "%1.1f%%", colors = colors)
    plt.legend(title = "Sentiments:", bbox_to_anchor = (1.05, 1), loc = 'upper left')
    file_name = col + "_pieplot.png"
    plt.savefig(file_name, dpi = 300)
    return plt


# In[13]:


def plotPie(labels, values):
    fig = go.Figure(
        go.Pie(
        labels = labels,
        values = [value*100 for value in values],
        hoverinfo = "label+percent",
        textinfo = "value"
    ))
    st.plotly_chart(fig, use_container_width=True)

lastSearched = ""
cacheData = {}


# ### User Interface and Dashboard using Streamlit

# In[14]:


def time_convert(sec):
    mins = int(sec//60)
    sec = round(sec%60, 2)
    hours = int(mins//60)
    time_lapsed = ''
    
    if hours == 0:
        if mins == 0:
            time_lapsed = str(sec) + 's'
        else:
            time_lapsed = str(mins) + 'm ' + str(sec) + 's'
    else:
        time_lapsed = str(hours) + 'h ' + str(mins) + 'm ' + str(sec) + 's'
    return time_lapsed


# In[17]:


def run():
    st.title("Twitter Sentiment Analysis")
    st.text("Analyze sentiments of a popular hashtag or topic on Twitter")
    
    search_item = st.text_input('Search Topic', placeholder = 'Input keyword HERE')

    placeholder = st.empty()
    btn = placeholder.button('Analyze', disabled = False, key = "1")

    if btn:
        if search_item:
            start_time = time.time()
        
            st.success('{}'.format(datetime.now().strftime("%d/%m/%Y %H:%M:%S")))
            
            placeholder.button('Analyze', disabled = True, key = "2")
            btn = st.empty()
            
            print('Analyzing..')

            data = scrape_tweets(search_item)
            print('data scraped..')

            cleaned_data = data_cleaning(data, search_item)
            print('data cleaned..')

            model_results = modeling(cleaned_data, search_item)
            print('classification completed..')

            st.subheader("Analysis of latest {} Tweets".format(len(model_results)))

            col1, col2 = st.columns(2)
            with col1:
                fig_cloud = create_cloud(model_results, search_item)
                print('wordcloud created')
                st.subheader("WordCloud of 100 words")
                st.pyplot(fig_cloud)
                
            with col2:
                fig = plt_segmentation('Label', model_results)
                print('segmentation plot created')
                st.subheader("Sentiment by Percent")
                st.pyplot(fig)

            pos_df = model_results[(model_results.Label == 'positive')].sort_values(by=['Score'], ascending = False)
            pos_df['censored'] = [profanity.censor(t) for t in pos_df.tweets]

            neg_df = model_results[(model_results.Label == 'negative')].sort_values(by=['Score'], ascending = False)
            neg_df['censored'] = [profanity.censor(t) for t in neg_df.tweets]

            col3, col4 = st.columns(2)
            with col3:
                st.subheader(":green[{} Positive Tweets]".format(len(pos_df)))
                for i in pos_df['censored'].head(3):
                    st.markdown(f'<h3 style="color:green;font-size:14px;">{i}</h3>', unsafe_allow_html=True)

            with col4:
                st.subheader(":red[{} Negative Tweets]".format(len(neg_df)))
                for i in neg_df['censored'].head(3):
                    st.markdown(f'<h3 style="color:red;font-size:14px;">{i}</h3>', unsafe_allow_html=True) 

            end_time = time.time()
            time_lapsed = time_convert(end_time - start_time)
            st.success('{}'.format(datetime.now().strftime("%d/%m/%Y %H:%M:%S")))
            
            st.success('Analysis finished in {}'.format(time_lapsed))
            print('Analysis is finished in {}'.format(time_lapsed))
            
            btn1 = st.button('Clear')
            
            if btn1:
                st.experimental_singleton.clear()
                run()
        else:
            st.warning("Please enter a keyword to analyze")

run()

