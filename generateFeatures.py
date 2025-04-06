import nltk
import pandas as pd
import numpy as np
import dill as pickle

from helpers import *
from ngram import NGram
from nltk import ngrams
from CountFeatureGenerator import *
from TfidfFeatureGenerator import *
from SvdFeatureGenerator import *
from Word2VecFeatureGenerator import *
from SentimentFeatureGenerator import *
from sklearn.model_selection import train_test_split


def print_gram(data, gram_type, print_grams):
    gram = gram_type + "gram"
    print("Sample" + gram + ": " + str(data["text_" + gram].iloc[0]) + "\n")
    if print_grams:
        print(data["text_" + gram])


# !!!Only one word for each keyword
def generate_grams(data, print_grams=False):
    print("Generating unigrams...")
    data["keyword_unigram"] = data["keyword"].map(lambda x: (list(nltk.word_tokenize(x))))
    data["text_unigram"] = data["text"].map(lambda x: (list(nltk.word_tokenize(x))))
    print_gram(data, "uni", print_grams)

    print("Generating bigrams...")
    data["keyword_bigram"] = data["keyword_unigram"].map(lambda x: [ ' '.join(grams) for grams in ngrams(x,2)])
    data["text_bigram"] = data["text_unigram"].map(lambda x: [ ' '.join(grams) for grams in ngrams(x,2)])
    print_gram(data, "bi", print_grams)

    print("Generating trigrams...")
    join_str = "_"
    data["keyword_trigram"] = data["keyword_unigram"].map(lambda x: [ ' '.join(grams) for grams in ngrams(x,3)])
    data["text_trigram"] = data["text_unigram"].map(lambda x: [ ' '.join(grams) for grams in ngrams(x,3)])
    print_gram(data, "tri", print_grams)

def process_data():
    full_data = pd.read_csv('./data/train.csv', encoding='utf-8')
    used_columns = ['keyword', 'text', 'location', 'target']
    full_data = full_data[used_columns].dropna()   #Drop all NAs or keep the rows but write NA?
    print("Loaded data shape:", full_data.shape)
    return full_data



def process():

    full_data = process_data()

    train = full_data.sample(frac=0.8, random_state=2025)
    test = full_data.loc[~full_data.index.isin(train.index)]
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)

    print('train.shape: ' + str(train.shape))
    n_train = train.shape[0]

    data = full_data
    test_flag = True

    if test_flag:
        print('data.shape: ' + str(data.shape))
        print(data)

        print('train.shape: ' + str(train.shape))
        print(train)

        print('test.shape: ' + str(test.shape))
        print(test)

    generate_grams(data, print_grams=False)

    with open('data.pkl', 'wb') as outfile:
        pickle.dump(data, outfile)
        print('dataframe saved in data.pkl')

    # define feature generators
    countFG    = CountFeatureGenerator()
    tfidfFG    = TfidfFeatureGenerator()
    svdFG      = SvdFeatureGenerator()
    word2vecFG = Word2VecFeatureGenerator()
    sentiFG    = SentimentFeatureGenerator()
    generators = [countFG, tfidfFG, svdFG, word2vecFG, sentiFG]

    for g in generators:
        g.process(data)

    for g in generators:
        g.read('train')
    for g in generators:
        g.read('test')

    print('done')

if __name__ == "__main__":
    process()