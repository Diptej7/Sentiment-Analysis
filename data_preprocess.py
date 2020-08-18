import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re


from nltk import SnowballStemmer
from nltk.corpus import stopwords

path = 'D:\\Projects\\Sentiment analysis\\data\\Main_data.csv'


data = pd.read_csv(path, encoding = 'latin',header=None)



data.columns = ['sentiment', 'id', 'date', 'query', 'user_id', 'text']

#drop useless columns
data = data.drop(['id', 'date', 'query', 'user_id'], axis=1)

#4 to positive, 2 to neutral and 0 to negative
lab_to_sentiment = {0:"Negative", 2:"Neutral", 4:"Positive"}
def label_decoder(label):
  return lab_to_sentiment[label]
data.sentiment = data.sentiment.apply(lambda x: label_decoder(x))

#check for skewness

val_count = data.sentiment.value_counts()

plt.figure(figsize=(8,4))
plt.bar(val_count.index, val_count.values)
plt.title("Sentiment Data Distribution")
#plt.show()

#clean text column

stop_words = stopwords.words('english')
stemmer = SnowballStemmer('english')

text_cleaning_re = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"

def preprocess(text, stem=False):
  text = re.sub(text_cleaning_re, ' ', str(text).lower()).strip()
  tokens = []
  for token in text.split():
    if token not in stop_words:
      if stem:
        tokens.append(stemmer.stem(token))
      else:
        tokens.append(token)
  return " ".join(tokens)
data.text = data.text.apply(lambda x: preprocess(x))

#store in new file
data.to_csv('D:\\Projects\\Sentiment analysis\\data\\final_data.csv')



#print(data[data['sentiment'].str.match('Negative')])
print(data.head(10))
