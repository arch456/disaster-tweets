#!/usr/bin/env python3

# Import libraries
import numpy as np
import pandas as pd
import re
from gensim.models.fasttext import FastText
from gensim.parsing.preprocessing import remove_stopwords
from gensim.parsing.preprocessing import strip_numeric
from gensim.parsing.preprocessing import strip_punctuation
from gensim.parsing.preprocessing import strip_multiple_whitespaces

DATA_DIR = 'data/'

train_file = DATA_DIR + 'train.csv'
test_file = DATA_DIR + 'test.csv'

def convert_to_numpy(df):
    print(df)

def load_test_data():
    return pd.read_csv(test_file, header=0)

def load_train_data():
    df = pd.read_csv(train_file, header=0)
    return df.iloc[:,:-1], df.iloc[:,-1]

# Load train and test data
if __name__ == "__main__":
    dataX, datay = load_train_data()
    print(dataX.head())
    print(datay.head())
    
# Compile url regex (a basic one)
url_regex = re.compile("(http|https)://[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+(/\S*)?")
# Compile regex to detect tokens that are entirely non-text
nontext_regex = re.compile("[^A-Za-z]+")
# Compile regex to detect @ mentions
mention_regex = re.compile("@\S+")
# Compile regex to detect various mis-encoded punctuation characters
misenc_regex = re.compile("&amp;")
# Compile regex to check if text is composed entirely of letters and digits
alphanum_regex = re.compile("[A-Za-z0-9]+")

sentencelist = []
#pre-process the text before applying FastText embeddings
for i in dataX['text']:
    text = url_regex.sub("", i)
    text = misenc_regex.sub("", text)
    text = mention_regex.sub("", text)
    text = re.sub("#", "", text)
    text = remove_stopwords(text)
    text = strip_numeric(text)
    text = strip_punctuation(text)
    text = strip_multiple_whitespaces(text)
    words = text.split(" ")
    sentencelist.append(words)

#build the FastText model and train
#use vector size 4 for the example
model = FastText(size=4, window=3, min_count=1)  # instantiate
model.build_vocab(sentences=sentencelist)
model.train(sentences=sentencelist, total_examples=len(sentencelist), epochs=10)  # train

#produce a list of word embeddings for the training data
veclist = []
for i in sentencelist[0:len(sentencelist)]:
    veclist.append(model.wv[i])
    
print(veclist[0])
print(len(veclist))
