import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

#caching the function if the input of the data doesn't change as each run will be using cpu a load_data
@st.cache(persist=True)

def load_data():
    data = pd.read_csv("/home/rhyme/Desktop/Project/Tweets.csv")
    data['tweet_created'] = pd.to_datetime(data['tweet_created'])
    return data

data  = load_data()

st.sidebar.subheader("Show random tweet")
random_tweet = st.sidebar.radio('Sentiment', ('positive', 'neutral', 'negative'))
st.sidebar.markdown(data.query('airline_sentiment == @random_tweet')[["text"]].sample(n=1).iat[0,0])
#.sample is used to return only text instead of dataframe and .iat ensure returns the text stored in 0th row and 0th column

st.sidebar.markdown("### Number of tweets  by sentiment")
select = st.sidebar.selectbox('Visualization type', ['Histogram', 'Pie Chart'], key='1')
sentiment_count = data['airline_sentiment'].value_counts()
sentiment_count = pd.DataFrame({'Sentiment':sentiment_count.index, 'Tweets':sentiment_count.values})

if not st.sidebar.checkbox("Hide", True):
    st.markdown("### Number of Tweets by sentiment")
    if select == "Histogram":
        fig = px.bar(sentiment_count, x='Sentiment', y='Tweets', color='Tweets', height=500)
        st.plotly_chart(fig)

    else:
        fig = px.pie(sentiment_count, values='Tweets', names='Sentiment')
        st.plotly_chart(fig)

st.sidebar.subheader("When and where are users tweeting from?")
hour = st.sidebar.slider("Hour of day", 0, 23) #description, min, max
#we can also do this by inputing time : st.sidebar.number_input("Hour of day", min_value=1, max_value=24)
modified_data = data[data['tweet_created'].dt.hour == hour] #dt.hour takes the data from the column tweet created which was converted to date time format and then used
if not st.sidebar.checkbox("Close", True, key='1'):
    st.markdown("### Tweet locations based on the time of day")
    st.markdown("%i tweets between %i:00 and %i:00" % (len(modified_data), hour, (hour+1)%24)) #text formatting
    st.map(modified_data)
    if st.sidebar.checkbox("Show raw data", False):
        st.write(modified_data)

st.sidebar.subheader("Breaking down sentiments by Tweets on airlines")
choice = st.sidebar.multiselect('Pick airlines:', ('US Airways', 'United', 'American', 'Southwest', 'Delta', 'Virgin America'), key='0')
if len(choice) > 0:
    choice_data = data[data.airline.isin(choice)]
    fig_choice = px.histogram(choice_data, x='airline', y='airline_sentiment', histfunc='count', color='airline_sentiment',
    facet_col='airline_sentiment', labels={'airline_sentiment', 'Tweets'}, height=600, width=800)
    st.plotly_chart(fig_choice)

#WordCloud using wordcloud import
st.sidebar.header("Word Cloud")
word_sentiment = st.sidebar.radio('Which sentiment do you need to display the word cloud for?', ('positive', 'neutral', 'negative'))

if not st.sidebar.checkbox("Close", True, key='3'):
    st.header('Word cloud for %s sentiment' % (word_sentiment))
    df = data[data['airline_sentiment']==word_sentiment]
    words = ' '.join(df['text'])
    processed_words = ' '.join([word for word in words.split() if 'http' not in word and not word.startswith('@') and word != 'RT'])
    wordcloud = WordCloud(stopwords=STOPWORDS, background_color='white', height=690, width=869).generate(processed_words)
    plt.imshow(wordcloud)
    plt.xticks([])
    plt.yticks([])
    st.pyplot()
